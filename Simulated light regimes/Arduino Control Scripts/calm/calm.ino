// Calm light regime: a steady point light source.
// Draws a filled circle of radius 0 (a single LED) at the centre of the 32 x 32 LED matrix,
// in white at full intensity.

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
  int center_x = 16;   // x coordinate of the light source
  int center_y = 16;   // y coordinate of the light source
  int radius = 0;      // radius of the light source [LEDs]; 0 = a single LED
  int intensity = 255; // intensity (0-255)

  matrix.fillCircle(center_x, center_y, radius, matrix.Color888(intensity, intensity, intensity));
}
