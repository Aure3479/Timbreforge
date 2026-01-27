#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLEClient.h>
#include <Arduino.h>

#include <Adafruit_NeoPixel.h>
#include <SoftwareSerial.h>

// ===================== VibraForge BLE UUIDs =====================
#define SERVICE_UUID        "f10016f6-542b-460a-ac8b-bbb0b2010599"
#define CHARACTERISTIC_UUID "f22535de-5375-44bd-8ca9-d0ea9ff9e410"

// ===================== Globals =====================
bool deviceConnected = false;

Adafruit_NeoPixel strip(1, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);

// Subchain UART TX pins (per your wiring)
const int subchain_pins[4] = {18, 17, 9, 8};
const int subchain_num = 4;

// VibraForge small units: 16 addresses per subchain (0..15)
static const int ADDRS_PER_SUBCHAIN = 16;

int global_counter = 0;

// ESP32 software serial UARTs (TX only: RX = -1)
EspSoftwareSerial::UART serial_group[4];

// --------------------- Helpers: NeoPixel ---------------------
static inline void neopixelPowerOn() {
#ifdef NEOPIXEL_POWER
  pinMode(NEOPIXEL_POWER, OUTPUT);
  digitalWrite(NEOPIXEL_POWER, HIGH);
#endif
}

static inline void setPixel(uint32_t c) {
  strip.setBrightness(20);
  strip.setPixelColor(0, c);
  strip.show();
}

// --------------------- Helpers: STOP ALL (safety) ---------------------
static void stopAllMotors() {
  // Send STOP command (one byte) to all motors on all subchains
  for (int sg = 0; sg < subchain_num; sg++) {
    for (int a = 0; a < ADDRS_PER_SUBCHAIN; a++) {
      uint8_t msg = (uint8_t)((a << 1) | 0); // stop = LSB 0
      serial_group[sg].write(&msg, 1);
      delayMicroseconds(200);
    }
  }
}

// ===================== BLE callbacks =====================
class MyCharacteristicCallbacks : public BLECharacteristicCallbacks {
public:
  void onWrite(BLECharacteristic *pCharacteristic) override {
    // IMPORTANT: binary-safe read
    std::string rxValue = pCharacteristic->getValue();
    const uint8_t* data = (const uint8_t*)rxValue.data();
    size_t len = rxValue.size();

    // We expect packets composed of 3-byte commands (padding allowed as 0xFF 0xFF 0xFF)
    if (len == 0) return;

    if (len % 3 != 0) {
      Serial.printf("Timestamp:%lu ms, Data=%u bytes, WRONG LENGTH\n",
                    (unsigned long)millis(), (unsigned)len);
      return;
    }

    Serial.printf("Timestamp:%lu ms, Data=%u bytes, #=%d\n",
                  (unsigned long)millis(), (unsigned)len, ++global_counter);

    for (size_t i = 0; i < len; i += 3) {
      uint8_t byte1 = data[i];
      uint8_t byte2 = data[i + 1];
      uint8_t byte3 = data[i + 2];

      if (byte1 == 0xFF) continue;  // padding command

      int serial_group_number = (byte1 >> 2) & 0x0F;
      int is_start = byte1 & 0x01;

      int addr = byte2 & 0x3F;            // 0..63 possible per spec, but typically 0..15 used
      int duty = (byte3 >> 3) & 0x0F;     // 0..15
      int freq = (byte3 >> 1) & 0x03;     // 0..3
      int wave = byte3 & 0x01;            // 0..1

      // Debug (optional)
      Serial.print("Received: ");
      Serial.print("SG: ");   Serial.print(serial_group_number);
      Serial.print(", Mode: "); Serial.print(is_start);
      Serial.print(", Addr: "); Serial.print(addr);
      Serial.print(", Duty: "); Serial.print(duty);
      Serial.print(", Freq: "); Serial.print(freq);
      Serial.print(", Wave: "); Serial.println(wave);

      sendCommand(serial_group_number, addr, is_start, duty, freq, wave);
    }
  }

private:
  void sendCommand(int serial_group_number, int motor_addr, int is_start,
                   int duty, int freq, int wave) {

    if (serial_group_number < 0 || serial_group_number >= subchain_num) return;
    if (motor_addr < 0 || motor_addr >= ADDRS_PER_SUBCHAIN) {
      // If you really use >15 per chain, remove this guard.
      // For VibraForge Small Unit, it’s typically 0..15 per chain.
      return;
    }

    if (is_start == 1) {
      // Start: 2 bytes
      uint8_t message[2];
      message[0] = (uint8_t)((motor_addr << 1) | 1);
      message[1] = (uint8_t)(0x80 | ((duty & 0x0F) << 3) | ((freq & 0x03) << 1) | (wave & 0x01));
      serial_group[serial_group_number].write(message, 2);
    } else {
      // Stop: 1 byte
      uint8_t message = (uint8_t)((motor_addr << 1) | 0);
      serial_group[serial_group_number].write(&message, 1);
    }
  }
};

class MyServerCallbacks : public BLEServerCallbacks {
public:
  void onConnect(BLEServer* pServer, esp_ble_gatts_cb_param_t *param) override {
    deviceConnected = true;

    Serial.println("connected!");
    char bda_str[18];
    sprintf(bda_str, "%02X:%02X:%02X:%02X:%02X:%02X",
            param->connect.remote_bda[0], param->connect.remote_bda[1],
            param->connect.remote_bda[2], param->connect.remote_bda[3],
            param->connect.remote_bda[4], param->connect.remote_bda[5]);
    Serial.println("Device connected with Address: " + String(bda_str));

    // Visual: blue when connected
    setPixel(strip.Color(0, 0, 255));
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("disconnected!");

    // SAFETY: stop everything on disconnect
    stopAllMotors();

    // Visual: green when advertising/ready
    setPixel(strip.Color(0, 255, 0));

    delay(200);
    BLEDevice::startAdvertising();
  }
};

// ===================== Setup / Loop =====================
void setup() {
  Serial.begin(500000);
  delay(200);

  Serial.print("number of hardware serial available: ");
  Serial.println(SOC_UART_NUM);

  // Init UART subchains (TX only)
  for (int i = 0; i < subchain_num; ++i) {
    Serial.print("initialize uart TX on pin ");
    Serial.println(subchain_pins[i]);

    serial_group[i].begin(115200, SWSERIAL_8E1, -1, subchain_pins[i], false);
    serial_group[i].enableIntTx(false);

    if (!serial_group[i]) {
      Serial.println("Invalid EspSoftwareSerial pin configuration, check config");
    }
    delay(100);
  }

  // Builtin LEDs (as in your original)
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);

  // NeoPixel
  neopixelPowerOn();
  strip.begin();
  setPixel(strip.Color(0, 255, 0)); // green = ready

  Serial.println("Starting BLE work!");

  // BLE setup
  BLEDevice::init("QT Py ESP32-S3");
  BLEDevice::setMTU(128);

  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  BLECharacteristic *pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE
  );

  pCharacteristic->setValue("0");
  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("Characteristic defined! Now you can read it in your phone!");
}

void loop() {
  // Nothing needed; BLE callbacks handle commands
  delay(10);
}
