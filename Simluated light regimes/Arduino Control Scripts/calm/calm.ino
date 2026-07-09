#include <RGBmatrixPanel.h>

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

  int center_x, center_y, radius, intensity, x_cord, y_cord, x_width, y_width;

  // Circle parameters - No longer used
  center_x = 16;
  center_y = 16;
  radius = 0.5;

  // Rectangle parameters - currently used
  intensity = 255; // 8 bit (255 color tones)
  x_cord = 0; 
  y_cord = 0;
  x_width = 10;
  y_width = 10;
  
   matrix.fillCircle(center_x, center_y, radius, matrix.Color888(intensity, intensity, intensity));
   //matrix.fillRect(x_cord, y_cord, x_width, y_width, matrix.Color888(intensity, intensity, intensity));
     
}
