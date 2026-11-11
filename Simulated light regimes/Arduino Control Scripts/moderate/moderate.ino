// Moderate light regime: a vertical bar of light sweeping across the 32 x 32 LED matrix.
// The bar moves one column every 20 ms and the sweep repeats continuously.

#include <RGBmatrixPanel.h>

// Pins of the LED matrix
#define CLK  8
#define OE   9
#define LAT 10
#define A   A0
#define B   A1
#define C   A2
#define D   A3

RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false);

void setup() {
  matrix.begin();
}

void loop() {
  int interval = 20;   // time per bar position [ms]
  int bar_width = 1;   // [LEDs]
  int bar_height = 32; // [LEDs], the full height of the matrix
  int intensity = 30;  // intensity (0-255)
  int shifts = 32 / bar_width;

  for (int i = 0; i < shifts + 1; i++) {
    matrix.fillRect(i * bar_width, 0, bar_width, bar_height, matrix.Color888(intensity, intensity, intensity));
    delay(interval);
    matrix.fillRect(0, 0, 32, 32, matrix.Color888(0, 0, 0)); // clear the matrix
  }
}
