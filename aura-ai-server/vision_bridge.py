"""
VisionBridge — Unified Pose + Gesture Recognition from a Single Frame Source
=============================================================================

Reads frames from either a webcam (index) or an ESP32-CAM MJPEG stream URL,
fans each frame out to both a Pose Landmarker and a Hand Landmarker, and
emits unified ActionTuple events that downstream code (ESP32Controller, etc.)
can consume without caring which model triggered them.

Output tuple schema:
    ActionTuple(source, device, action)
    e.g. ("gesture", "fan",   "toggle")
         ("pose",    "light",  "on")
         ("pose",    "ac",     "off")

Sources
-------
- "gesture" : deliberate finger-count command (user raised N fingers)
- "pose"    : ambient posture change crossed a probability threshold

Usage (standalone)
------------------
    from vision_bridge import VisionBridge, ActionTuple

    bridge = VisionBridge(
        source="http://192.168.1.50:81/stream",   # or 0 for webcam
        pose_model_path="pose_landmarker.task",
        hand_model_path="hand_landmarker.task",
    )

    for frame, actions in bridge.run():
        for action in actions:
            print(action)               # ActionTuple(source='gesture', device='fan', action='toggle')
            # -> send to ESP32, SpacyParser, etc.
        cv2.imshow("Bridge", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Or drive it yourself frame-by-frame:
    bridge.open()
    while True:
        frame, actions = bridge.step()
        for a in actions:
            print(a)
        cv2.imshow("Bridge", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    bridge.close()

Changes vs original
-------------------
BUG FIX  1 — _GestureRecogniser._last_gesture_type stored str(count) but
             GESTURE_MAP keys are int, so the "same gesture held" dedup
             never fired.  Now stores the raw int in _last_gesture_count.

BUG FIX  2 — _GestureRecogniser.draw() and _PoseRecogniser.draw() were
             never called from step(); they took a MediaPipe `result`
             object that step() did not retain.  Both classes now cache
             their last result so draw() works without the caller passing
             anything.  step() calls them when show_overlay is True.

BUG FIX  3 — _PoseRecogniser._classify() used knee-landmark y-coords
             without checking MediaPipe's per-landmark visibility score.
             Low-visibility knees return noisy coordinates that push the
             heuristic into LYING_DOWN / ABSENT incorrectly.  The method
             now gates knee-dependent paths on
             min(lm[25].visibility, lm[26].visibility) >= 0.4.

BUG FIX  4 — WEIGHTS table only defined "light" and "fan" but the module
             docstring listed ac and tv as well.  WEIGHTS and _states are
             now consistent: light, fan, ac, tv.  Posture weights for the
             two new appliances follow the original design intent.

IMPROVEMENT 1 — run() retried exactly once on a dropped frame, then gave
                up.  For ESP32-CAM streams, network blips are common.
                Added configurable max_retries (default 5) with
                exponential back-off capped at 2 s.

IMPROVEMENT 2 — VisionBridge.__init__ and open() wrapped model
                initialisation in try/except so a bad .task path raises a
                clear Python RuntimeError instead of a raw C++ exception
                from MediaPipe.

IMPROVEMENT 3 — _draw_overlay now calls the cached draw helpers so
                landmarks are rendered on the frame alongside the HUD.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionTuple:
    """
    A single resolved action emitted by the bridge.

    Attributes
    ----------
    source : "gesture" | "pose"
        Which model produced this action.
    device : str
        Target appliance name ("light", "fan", "ac", "tv", "all").
    action : str
        What to do ("on", "off", "toggle").
    confidence : float
        0-1. For gesture: finger-count certainty (always 1.0).
        For pose: appliance probability that crossed the threshold.
    """
    source: str
    device: str
    action: str
    confidence: float = 1.0

    def as_tuple(self) -> Tuple[str, str, str]:
        """Return (source, device, action) — mirrors SpacyParser's (device, action) style."""
        return (self.source, self.device, self.action)


# ---------------------------------------------------------------------------
# Gesture recogniser (Hand Landmarker — IMAGE mode)
# ---------------------------------------------------------------------------

class _GestureRecogniser:
    """
    Wraps MediaPipe Hand Landmarker in IMAGE running mode.

    Finger-count -> (device, action) mapping:
        0 fingers (fist)  -> light OFF
        1 finger          -> fan OFF
        2 fingers         -> fan ON
        5 fingers (open)  -> light ON

    Uses angle-based thumb detection so left/right hands are handled correctly.
    """

    GESTURE_MAP: Dict[int, Tuple[str, str]] = {
        0: ("light", "off"),
        1: ("fan",   "off"),
        2: ("fan",   "on"),
        5: ("light", "on"),
    }

    def __init__(
        self,
        model_path: str,
        cooldown: float = 1.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.cooldown = cooldown
        self._last_gesture_time = 0.0
        # FIX 1: store the raw int count, not str(count), so comparison works
        self._last_gesture_count: Optional[int] = None
        self._hand_absent_since: float = 0.0
        self._appliance_states: Dict[str, Optional[str]] = {}

        # Cache last MediaPipe result so draw() can be called without the
        # caller re-running detection.  (FIX 2)
        self._last_result = None

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hand Landmarker from '{model_path}'. "
                "Check the path and that the file is a valid .task bundle."
            ) from exc

    def close(self) -> None:
        self._landmarker.close()

    def update_state(self, device: str, action: str) -> None:
        """
        Inform the gesture recogniser of an external state change
        (e.g. from voice command or pose model) so it won't re-emit
        the same action unnecessarily.
        """
        self._appliance_states[device] = action

    def process(self, frame_bgr: np.ndarray) -> Optional[ActionTuple]:
        """
        Run hand detection on one BGR frame.
        Returns an ActionTuple if a new gesture is confirmed, else None.
        """
        now = time.time()
        if now - self._last_gesture_time < self.cooldown:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        # FIX 2: cache result for draw()
        self._last_result = result

        if not result.hand_landmarks:
            if self._hand_absent_since == 0.0:
                self._hand_absent_since = now
            elif now - self._hand_absent_since >= self.cooldown:
                self._last_gesture_count = None   # FIX 1: reset int, not str
                self._hand_absent_since = 0.0
            return None

        self._hand_absent_since = 0.0

        landmarks = result.hand_landmarks[0]
        handedness = (
            result.handedness[0][0].category_name
            if result.handedness and result.handedness[0]
            else None
        )

        count = self._count_fingers(landmarks, handedness)

        # FIX 1: compare int to int — this check now actually fires
        if count == self._last_gesture_count:
            return None

        if count not in self.GESTURE_MAP:
            # Still update so we don't re-fire after an unrecognised count
            self._last_gesture_count = count
            return None

        device, action = self.GESTURE_MAP[count]

        if self._appliance_states.get(device) == action:
            self._last_gesture_time = now
            self._last_gesture_count = count   # FIX 1
            return None

        self._last_gesture_time = now
        self._last_gesture_count = count       # FIX 1
        self._appliance_states[device] = action

        return ActionTuple(source="gesture", device=device, action=action, confidence=1.0)

    # ------------------------------------------------------------------
    # Landmark drawing helper
    # ------------------------------------------------------------------

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw hand skeleton using the last cached MediaPipe result."""
        # FIX 2: use cached result; no longer requires caller to pass it
        result = self._last_result
        if result is None or not result.hand_landmarks:
            return frame_bgr
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,17),(5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
        ]
        h, w = frame_bgr.shape[:2]
        for hand in result.hand_landmarks:
            for s, e in CONNECTIONS:
                p1 = (int(hand[s].x * w), int(hand[s].y * h))
                p2 = (int(hand[e].x * w), int(hand[e].y * h))
                cv2.line(frame_bgr, p1, p2, (0, 255, 0), 2)
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)
        return frame_bgr

    # ------------------------------------------------------------------
    # Internal: finger counting
    # ------------------------------------------------------------------

    @staticmethod
    def _angle(a, b, c) -> float:
        """Angle at point b formed by a-b-c."""
        va = np.array([a.x - b.x, a.y - b.y])
        vc = np.array([c.x - b.x, c.y - b.y])
        na, nc = np.linalg.norm(va), np.linalg.norm(vc)
        if na == 0 or nc == 0:
            return 0.0
        cos = np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def _count_fingers(self, lm: list, handedness: Optional[str]) -> int:
        """Angle + distance based finger counting; handles left/right hands."""
        thumb_angle = self._angle(lm[4], lm[3], lm[2])
        tip_to_base = np.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y)
        joint_to_base = np.hypot(lm[3].x - lm[5].x, lm[3].y - lm[5].y)

        if handedness == "Right":
            thumb_dir = lm[4].x < lm[3].x
        elif handedness == "Left":
            thumb_dir = lm[4].x > lm[3].x
        else:
            thumb_dir = abs(lm[4].x - lm[3].x) > 0.02

        four_fingers_up = (
            lm[8].y  < lm[6].y  < lm[5].y and
            lm[12].y < lm[10].y < lm[9].y and
            lm[16].y < lm[14].y < lm[13].y and
            lm[20].y < lm[18].y < lm[17].y
        )
        if four_fingers_up:
            thumb_up = thumb_angle > 130 and thumb_dir
        else:
            thumb_up = thumb_angle > 150 and tip_to_base > joint_to_base * 1.15 and thumb_dir

        states = {
            "thumb":  thumb_up,
            "index":  lm[8].y  < lm[6].y  < lm[5].y,
            "middle": lm[12].y < lm[10].y < lm[9].y,
            "ring":   lm[16].y < lm[14].y < lm[13].y,
            "pinky":  lm[20].y < lm[18].y < lm[17].y,
        }
        return sum(states.values())


# ---------------------------------------------------------------------------
# Pose recogniser (Pose Landmarker — VIDEO mode)
# ---------------------------------------------------------------------------

class _PoseRecogniser:
    """
    Wraps MediaPipe Pose Landmarker in VIDEO running mode.

    Posture -> appliance probability mapping:
        STANDING   -> lights ON,  fan ON,  ac partial,  tv OFF
        SITTING    -> lights ON,  fan partial, ac ON,   tv ON
        LYING_DOWN -> lights OFF, fan OFF, ac partial,  tv partial
        ABSENT     -> everything OFF

    Emits ActionTuples only when an appliance probability crosses a threshold
    AND that state differs from the last confirmed state (hysteresis).
    """

    # FIX 4: WEIGHTS now covers all four appliances declared in the docstring.
    # Posture weights follow original design intent:
    #   ac: high when sitting (desk work) or lying (sleep cooling), partial standing
    #   tv: high when sitting or lying, off when standing
    WEIGHTS: Dict[str, Dict[str, Tuple[float, float]]] = {
        "light": {
            "STANDING":   (0.90, 0.00),
            "SITTING":    (0.80, 0.00),
            "LYING_DOWN": (0.00, 0.90),
            "ABSENT":     (0.00, 1.00),
        },
        "fan": {
            "STANDING":   (0.85, 0.00),
            "SITTING":    (0.30, 0.00),
            "LYING_DOWN": (0.00, 0.85),
            "ABSENT":     (0.00, 1.00),
        },
        "ac": {
            "STANDING":   (0.40, 0.00),
            "SITTING":    (0.80, 0.00),
            "LYING_DOWN": (0.65, 0.00),
            "ABSENT":     (0.00, 1.00),
        },
        "tv": {
            "STANDING":   (0.00, 0.80),
            "SITTING":    (0.85, 0.00),
            "LYING_DOWN": (0.60, 0.00),
            "ABSENT":     (0.00, 1.00),
        },
    }

    # Minimum per-landmark visibility score to trust knee coordinates.
    # Below this, knee-dependent posture paths fall back to upper-body only.
    KNEE_VISIBILITY_MIN = 0.4

    ON_THRESHOLD  = 0.70
    OFF_THRESHOLD = 0.25

    def __init__(
        self,
        model_path: str,
        on_threshold:  float = 0.70,
        off_threshold: float = 0.25,
        min_confidence: float = 0.5,
        fps: float = 30.0,
        lying_confirm_secs: float = 3.0,
    ):
        self.on_threshold  = on_threshold
        self.off_threshold = off_threshold
        self._fps = fps
        self._frame_count = 0
        self.lying_confirm_secs = lying_confirm_secs

        # FIX 4: initialise states for all four devices
        self._states: Dict[str, Optional[bool]] = {d: None for d in self.WEIGHTS}
        self._pending: Dict[str, Optional[tuple]] = {d: None for d in self.WEIGHTS}

        # FIX 2: cache last result for draw()
        self._last_result = None

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Pose Landmarker from '{model_path}'. "
                "Check the path and that the file is a valid .task bundle."
            ) from exc

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray) -> List[ActionTuple]:
        """
        Run pose detection on one BGR frame.
        Returns a (possibly empty) list of ActionTuples for state changes.

        NOTE: Must be called for EVERY frame (VIDEO mode requires monotonically
        increasing timestamps — skipping frames breaks the tracker).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self._frame_count * 1000 / self._fps)
        self._frame_count += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        # FIX 2: cache for draw()
        self._last_result = result

        action_probs = self._classify(result)
        appliance_probs = self._appliance_probs(action_probs)

        now = time.time()
        posture = max(action_probs, key=action_probs.get)
        actions: List[ActionTuple] = []

        for device, prob in appliance_probs.items():
            confirmed = self._states[device]

            if prob >= self.on_threshold:
                desired = True
            elif prob <= self.off_threshold:
                desired = False
            else:
                desired = confirmed

            if desired is None:
                continue

            if desired == confirmed:
                self._pending[device] = None
                continue

            if desired is False and posture == "LYING_DOWN":
                required = self.lying_confirm_secs
            elif desired is True:
                required = 0.5
            else:
                required = 1.0

            pending = self._pending[device]

            if pending is None or pending[0] != desired:
                self._pending[device] = (desired, now)
            elif (now - pending[1]) >= required:
                self._states[device] = desired
                self._pending[device] = None
                actions.append(ActionTuple(
                    source="pose",
                    device=device,
                    action="on" if desired else "off",
                    confidence=round(prob, 3),
                ))

        return actions

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw pose skeleton using the last cached MediaPipe result."""
        # FIX 2: use cached result
        result = self._last_result
        if result is None or not result.pose_landmarks:
            return frame_bgr
        CONNECTIONS = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
        ]
        h, w = frame_bgr.shape[:2]
        for lm_list in result.pose_landmarks:
            for s, e in CONNECTIONS:
                if s < len(lm_list) and e < len(lm_list):
                    p1 = (int(lm_list[s].x * w), int(lm_list[s].y * h))
                    p2 = (int(lm_list[e].x * w), int(lm_list[e].y * h))
                    cv2.line(frame_bgr, p1, p2, (255, 200, 0), 2)
        return frame_bgr

    # ------------------------------------------------------------------
    # Internal: posture classification & appliance probability
    # ------------------------------------------------------------------

    def _classify(self, result) -> Dict[str, float]:
        """
        Classify posture from pose landmarks.
        Returns probability dict across STANDING/SITTING/LYING_DOWN/ABSENT.

        FIX 3: knee coordinates are gated on landmark visibility score.
        When knees are not reliably visible (side-on, far from camera,
        partially occluded) the method falls back to an upper-body-only
        heuristic rather than using noisy extrapolated coordinates.
        """
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return {"STANDING": 0.0, "SITTING": 0.0, "LYING_DOWN": 0.0, "ABSENT": 1.0}

        lm = result.pose_landmarks[0]

        try:
            shoulder_y = (lm[11].y + lm[12].y) / 2
            hip_y      = (lm[23].y + lm[24].y) / 2

            lean = hip_y - shoulder_y

            # FIX 3: check visibility of knee landmarks before using them
            left_knee_vis  = getattr(lm[25], "visibility", 0.0)
            right_knee_vis = getattr(lm[26], "visibility", 0.0)
            knees_reliable = min(left_knee_vis, right_knee_vis) >= self.KNEE_VISIBILITY_MIN

            if knees_reliable:
                knee_y     = (lm[25].y + lm[26].y) / 2
                body_height = abs(knee_y - shoulder_y)
                knee_bend   = knee_y - hip_y

                if lean > 0.2 and knee_bend > 0.1:
                    probs = {"STANDING": 0.85, "SITTING": 0.10, "LYING_DOWN": 0.05, "ABSENT": 0.0}
                elif lean > 0.05 and knee_bend < 0.08:
                    probs = {"STANDING": 0.10, "SITTING": 0.80, "LYING_DOWN": 0.10, "ABSENT": 0.0}
                elif body_height < 0.15:
                    probs = {"STANDING": 0.05, "SITTING": 0.10, "LYING_DOWN": 0.85, "ABSENT": 0.0}
                else:
                    probs = {"STANDING": 0.33, "SITTING": 0.34, "LYING_DOWN": 0.33, "ABSENT": 0.0}

            else:
                # Upper-body fallback: lean alone determines standing vs sitting;
                # can't distinguish lying down without reliable knee positions so
                # we spread probability rather than guess LYING_DOWN confidently.
                logger.debug(
                    "Knee landmarks low visibility (L=%.2f R=%.2f); "
                    "falling back to upper-body posture heuristic.",
                    left_knee_vis, right_knee_vis,
                )
                if lean > 0.2:
                    probs = {"STANDING": 0.75, "SITTING": 0.15, "LYING_DOWN": 0.10, "ABSENT": 0.0}
                elif lean > 0.05:
                    probs = {"STANDING": 0.15, "SITTING": 0.65, "LYING_DOWN": 0.20, "ABSENT": 0.0}
                else:
                    probs = {"STANDING": 0.25, "SITTING": 0.25, "LYING_DOWN": 0.25, "ABSENT": 0.25}

        except (IndexError, AttributeError):
            probs = {"STANDING": 0.0, "SITTING": 0.0, "LYING_DOWN": 0.0, "ABSENT": 1.0}

        return probs

    def _appliance_probs(self, action_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Convert posture probabilities to per-appliance ON probability using
        the WEIGHTS table (dot product of posture probs and on-weights).
        """
        result = {}
        for device, posture_weights in self.WEIGHTS.items():
            on_prob = sum(
                action_probs.get(posture, 0.0) * posture_weights[posture][0]
                for posture in posture_weights
            )
            result[device] = round(on_prob, 3)
        return result


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class VisionBridge:
    """
    Single entry point that fans one frame source into both pose and gesture
    recognition and emits unified ActionTuple events.

    Parameters
    ----------
    source : int | str
        Camera index (0 for webcam) or MJPEG stream URL
        (e.g. "http://192.168.1.50:81/stream").
    pose_model_path : str
        Path to pose_landmarker.task
    hand_model_path : str
        Path to hand_landmarker.task
    gesture_cooldown : float
        Seconds between accepted gesture events (default 1.0).
    pose_on_threshold : float
        Appliance probability to trigger ON (default 0.70).
    pose_off_threshold : float
        Appliance probability to trigger OFF (default 0.25).
    lying_confirm_secs : float
        Seconds a lying-down posture must be held before triggering
        appliance OFF (default 3.0).
    min_confidence : float
        Shared detection confidence floor for both models (default 0.5).
    fps_hint : float
        Used for VIDEO-mode timestamps. Set to your stream's actual FPS
        (default 30). Overridden automatically by the real capture FPS
        when the source is opened.
    show_overlay : bool
        Draw landmarks + action text on frames returned by step()
        (default True).
    max_retries : int
        How many consecutive frame-read failures to tolerate in run()
        before giving up. Uses exponential back-off capped at 2 s.
        (default 5, IMPROVEMENT 1)
    """

    def __init__(
        self,
        source: Union[int, str],
        pose_model_path: str,
        hand_model_path: str,
        gesture_cooldown: float = 1.0,
        pose_on_threshold: float = 0.70,
        pose_off_threshold: float = 0.25,
        lying_confirm_secs: float = 3.0,
        min_confidence: float = 0.5,
        fps_hint: float = 30.0,
        show_overlay: bool = True,
        max_retries: int = 5,
    ):
        self.source = source
        self.show_overlay = show_overlay
        self.max_retries = max_retries

        # IMPROVEMENT 2: model init errors now surface as clear RuntimeErrors
        self._gesture = _GestureRecogniser(
            model_path=hand_model_path,
            cooldown=gesture_cooldown,
            min_detection_confidence=min_confidence,
        )
        self._pose = _PoseRecogniser(
            model_path=pose_model_path,
            on_threshold=pose_on_threshold,
            off_threshold=pose_off_threshold,
            min_confidence=min_confidence,
            fps=fps_hint,
            lying_confirm_secs=lying_confirm_secs,
        )

        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video source. Call before step()."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"VisionBridge: could not open source '{self.source}'.\n"
                "Check webcam index or ESP32-CAM stream URL."
            )
        real_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if real_fps and real_fps > 0:
            self._pose._fps = real_fps

    def close(self) -> None:
        """Release all resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._gesture.close()
        self._pose.close()
        cv2.destroyAllWindows()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Frame-level API
    # ------------------------------------------------------------------

    def step(self) -> Tuple[Optional[np.ndarray], List[ActionTuple]]:
        """
        Read one frame and run both models.

        Returns
        -------
        (frame, actions)
            frame   : BGR ndarray with optional overlay drawn on it.
                      None if the frame could not be read.
            actions : list of ActionTuple (may be empty).

        NOTE: Pose model uses VIDEO mode — you MUST call step() for every
        frame, even if you don't use the result, to keep timestamps monotonic.
        """
        if self._cap is None:
            raise RuntimeError("Call open() before step().")

        ok, frame = self._cap.read()
        if not ok:
            return None, []

        if isinstance(self.source, int):
            frame = cv2.flip(frame, 1)

        actions: List[ActionTuple] = []

        gesture_action = self._gesture.process(frame)
        if gesture_action:
            actions.append(gesture_action)

        pose_actions = self._pose.process(frame)
        actions.extend(pose_actions)

        for pa in pose_actions:
            self._gesture.update_state(pa.device, pa.action)

        # IMPROVEMENT 3: draw() now uses cached results; no separate result arg needed
        if self.show_overlay:
            frame = self._gesture.draw(frame)
            frame = self._pose.draw(frame)
            frame = self._draw_overlay(frame, actions)

        return frame, actions

    # ------------------------------------------------------------------
    # Generator API
    # ------------------------------------------------------------------

    def run(self) -> Generator[Tuple[Optional[np.ndarray], List[ActionTuple]], None, None]:
        """
        Generator that yields (frame, actions) until the source is exhausted
        or the bridge is closed.

        IMPROVEMENT 1: configurable retry loop with exponential back-off
        instead of a single retry.  Tolerates transient network blips from
        ESP32-CAM streams.

        Usage
        -----
        with VisionBridge(...) as bridge:
            for frame, actions in bridge.run():
                for action in actions:
                    print(action.as_tuple())
                cv2.imshow("Bridge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        """
        if self._cap is None:
            self.open()

        consecutive_failures = 0

        while True:
            frame, actions = self.step()

            if frame is None:
                consecutive_failures += 1
                if consecutive_failures > self.max_retries:
                    logger.error(
                        "VisionBridge: %d consecutive frame failures — giving up.",
                        consecutive_failures,
                    )
                    break
                # Exponential back-off: 0.05, 0.1, 0.2, 0.4, ... capped at 2 s
                wait = min(0.05 * (2 ** (consecutive_failures - 1)), 2.0)
                logger.warning(
                    "VisionBridge: frame read failed (attempt %d/%d), "
                    "retrying in %.2f s.",
                    consecutive_failures, self.max_retries, wait,
                )
                time.sleep(wait)
                continue

            consecutive_failures = 0
            yield frame, actions

    # ------------------------------------------------------------------
    # Overlay drawing (HUD only — landmarks drawn by model helpers above)
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame: np.ndarray, actions: List[ActionTuple]) -> np.ndarray:
        h, w = frame.shape[:2]

        # Pose device states (top-right panel)
        x, y = max(10, w - 230), 30
        cv2.putText(frame, "APPLIANCES", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
        # FIX 4: iterates over all four devices now
        for device, state_val in self._pose._states.items():
            state_str = "ON" if state_val else ("OFF" if state_val is False else "?")
            color = (0, 255, 0) if state_val else (120, 120, 120)
            cv2.putText(frame, f"{device.upper()}: {state_str}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y += 18

        if actions:
            latest = actions[-1]
            label = f"[{latest.source.upper()}] {latest.device.upper()} {latest.action.upper()}"
            color = (0, 255, 0) if latest.action in ("on", "toggle") else (0, 0, 255)
            cv2.putText(frame, label, (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        hint = "fist=light off | 5=light on | 1=fan off | 2=fan on"
        cv2.putText(frame, hint, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return frame


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="VisionBridge: unified pose + gesture control")
    p.add_argument("--source", default="0",
                   help="Camera index (0) or ESP32-CAM stream URL")
    p.add_argument("--pose-model",  required=True, help="Path to pose_landmarker.task")
    p.add_argument("--hand-model",  required=True, help="Path to hand_landmarker.task")
    p.add_argument("--gesture-cooldown", type=float, default=1.0)
    p.add_argument("--on-threshold",     type=float, default=0.70)
    p.add_argument("--off-threshold",    type=float, default=0.25)
    p.add_argument("--max-retries",      type=int,   default=5)
    p.add_argument("--no-overlay", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    bridge = VisionBridge(
        source=source,
        pose_model_path=args.pose_model,
        hand_model_path=args.hand_model,
        gesture_cooldown=args.gesture_cooldown,
        pose_on_threshold=args.on_threshold,
        pose_off_threshold=args.off_threshold,
        show_overlay=not args.no_overlay,
        max_retries=args.max_retries,
    )

    print("VisionBridge started. Press 'q' to quit.")
    with bridge:
        for frame, actions in bridge.run():
            for action in actions:
                print(action.as_tuple())
            if frame is not None:
                cv2.imshow("VisionBridge", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()