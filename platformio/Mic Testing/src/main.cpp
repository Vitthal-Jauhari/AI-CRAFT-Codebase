/////////////////////////////////////////////////////////////////
/*
  Broadcasting Your Voice with ESP32-S3 & INMP441
  For More Information: https://youtu.be/qq2FRv0lCPw
  Created by Eric N. (ThatProject)
*/
/////////////////////////////////////////////////////////////////

/*
- Device
ESP32-S3 DevKit-C
https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html

- Required Library
Arduino ESP32: 2.0.9

Arduino Websockets: 0.5.3
https://github.com/gilmaimon/ArduinoWebsockets
*/

#include <driver/i2s.h>
#include <WiFi.h>
#include <ArduinoWebsockets.h>

// ---- Function Prototypes ----
void connectWiFi();
void connectWSServer();
void micTask(void *parameter);

#define I2S_SD 33
#define I2S_WS 25
#define I2S_SCK 26
#define I2S_PORT I2S_NUM_0

#define bufferCnt 10
#define bufferLen 1024

const char *ssid = "OnePlus Nord 3 5G";
const char *password = "12345678";

const char *websocket_server_host = "10.134.10.151";
const uint16_t websocket_server_port = 42069; // <WEBSOCKET_SERVER_PORT>

using namespace websockets;
WebsocketsClient client;
bool isWebSocketConnected;

void onEventsCallback(WebsocketsEvent event, String data)
{
    if (event == WebsocketsEvent::ConnectionOpened)
    {
        Serial.println("Connnection Opened");
        isWebSocketConnected = true;
    }
    else if (event == WebsocketsEvent::ConnectionClosed)
    {
        Serial.println("Connnection Closed");
        isWebSocketConnected = false;
    }
    else if (event == WebsocketsEvent::GotPing)
    {
        Serial.println("Got a Ping!");
    }
    else if (event == WebsocketsEvent::GotPong)
    {
        Serial.println("Got a Pong!");
    }
}

void i2s_install()
{

    const i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 44100,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
    };

    /**
     * This was code used before
     */

    // Set up I2S Processor configuration
    // const i2s_config_t i2s_config = {
    //     .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    //     .sample_rate = 44100,
    //     //.sample_rate = 16000,
    //     .bits_per_sample = i2s_bits_per_sample_t(16),
    //     .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    //     .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    //     .intr_alloc_flags = 0,
    //     .dma_buf_count = bufferCnt,
    //     .dma_buf_len = bufferLen,
    //     .use_apll = false};

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin()
{
    const i2s_pin_config_t pin_config = {
        .bck_io_num = 26,
        .ws_io_num = 25,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = 33};

    /**
     * This was code used before
     */

    // Set I2S pin configuration
    // const i2s_pin_config_t pin_config = {
    //     .bck_io_num = I2S_SCK,
    //     .ws_io_num = I2S_WS,
    //     .data_out_num = -1,
    //     .data_in_num = I2S_SD};

    i2s_set_pin(I2S_PORT, &pin_config);
}

void setup()
{
    Serial.begin(115200);

    connectWiFi();
    connectWSServer();
    xTaskCreatePinnedToCore(micTask, "micTask", 10000, NULL, 1, NULL, 1);
}

void loop()
{
}

void connectWiFi()
{
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
}

void connectWSServer()
{
    client.onEvent(onEventsCallback);
    while (!client.connect(websocket_server_host, websocket_server_port, "/"))
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("Websocket Connected!");
}

int32_t audioBuffer[bufferLen];

void micTask(void *parameter)
{

    i2s_install();
    i2s_setpin();
    i2s_start(I2S_PORT);

    size_t bytesIn = 0;

    while (1)
    {
        esp_err_t result = i2s_read(
            I2S_PORT,
            audioBuffer,
            bufferLen * sizeof(int32_t),
            &bytesIn,
            portMAX_DELAY);

        if (result == ESP_OK && isWebSocketConnected)
        {
            client.sendBinary((const char *)audioBuffer, bytesIn);
        }
    }
}

// void micTask(void *parameter)
// {

//     i2s_install();
//     i2s_setpin();
//     i2s_start(I2S_PORT);

//     size_t bytesIn = 0;
//     while (1)
//     {
//         esp_err_t result = i2s_read(I2S_PORT, &sBuffer, bufferLen, &bytesIn, portMAX_DELAY);
//         if (result == ESP_OK && isWebSocketConnected)
//         {
//             client.sendBinary((const char *)sBuffer, bytesIn);
//         }
//     }
// }