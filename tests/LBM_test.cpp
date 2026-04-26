// THIS CODE IS A QUICK ONE FILE CODE FOR OPTIMISED LBM METHOD INSPIRED BY 
//https://vanhunteradams.com/DE1/Lattice_Boltzmann/Lattice_Boltzmann.html#Python/C-representation
// its architecture is optimised for fast computing and was initially coded in c for D2Q9. 

#include <cmath>
#include <vector>
#include <GLFW/glfw3.h>
#include <iostream>

int height = 232;                      // grid height
int width = 512;                      // grid width
float viscosity = 0.02f;                // viscosity
float omega = 1.0f/(3.0f *viscosity + 0.5f);   // relaxation parameter (a function of viscosity)
float u0 = 0.1f;                    // initial in-flow speed (eastward)
float four9ths = 4.0f/9.0f;              // a constant
float one9th   = 1.0f/9.0f ;              // a constant
float one36th  = 1.0f/36.0f  ;             // a constant

// Microscopic densities
std::vector<float> n0(height*width, 0.0f);  // Naught
std::vector<float> nN(height*width, 0.0f);  // North
std::vector<float> nS(height*width, 0.0f);  // South
std::vector<float> nE(height*width, 0.0f);  // East
std::vector<float> nW(height*width, 0.0f); // West
std::vector<float> nNW(height*width, 0.0f); // Northwest
std::vector<float> nNE(height*width, 0.0f); // Northeast
std::vector<float> nSE(height*width, 0.0f); // Southeast
std::vector<float> nSW(height*width, 0.0f); // Southwest

// Barriers
std::vector<float> bar(height*width, 0.0f);  // Barriers

// Macroscopic density and velocity
std::vector<float> rho(height*width, 0.0f);   // Cell density
std::vector<float> ux(height*width, 0.0f);    // Cell x-velocity
std::vector<float> uy(height*width, 0.0f);   // Cell y-velocity
std::vector<float> speed2(height*width, 0.0f); // Cell squared velocity



void initialize(int xtop, int ytop, int yheight, float u0) {
	
    int xcoord = 0;
    int ycoord = 0;
    
    int count = 0;
     for(int i =0; i<height*width ; i++){

        n0[i] = four9ths* (1 - 1.5*(u0*u0));
        nN[i] = one9th  * (1 - 1.5*(u0*u0));
        nS[i] = one9th  * (1 - 1.5*(u0*u0));
        nE[i] = one9th  * (1 + 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        nW[i] = one9th  * (1 - 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        nNE[i]= one36th * (1 + 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        nSE[i]= one36th * (1 + 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        nNW[i]= one36th * (1 - 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        nSW[i]= one36th * (1 - 3*u0 + 4.5*(u0*u0) - 1.5*(u0*u0));
        
        if (xcoord == xtop) {
			if (ycoord >= ytop) {
				if (ycoord < (ytop+yheight)) {
					bar[ ycoord*width + xcoord] = 1 ;
				}
			}
		}

		xcoord = (xcoord < (width-1)) ? (xcoord+1) : 0 ;
		ycoord = (xcoord != 0) ? ycoord : (ycoord+1) ;
            
    }
}


void stream(){
    int i;
    int j;
    for(i =0; i<width-1 ; i++){
        for(j =0; j<height-1 ; j++){

            nN[i + width*j] = nN[i + width*j + width];
            nNW[i + width*j] = nNW[i + width*j + width + 1];
            nW[i + width*j] = nW[i + width*j + 1];
            nS[i + (height-1-j)*width] = nS[i + (height-2-j)*width];
            nSW[i + (height-1-j)*width] = nSW[i + (height-2-j)*width + 1];
            nE[(width - 1 - i) + width*j] = nE[(width - 2 - i) + width*j ];
            nNE[(width - 1 - i) + width*j] = nNE[(width - 2 - i) + width*j + width];
            nSE[(width - 1 - i) + (height-1-j)*width] = nSE[(width - 2 - i) + (height-2-j)*width];
        }

    }

    i += 1 ;
    for (j=1; j<(height-1); j++) {
        // Movement north on right boundary (Northwest corner)
        nN[j*width + i] = nN[j*width + i + width];
        // Movement south on right boundary (Southwest corner)
        nS[(height-j-1)*width + i] = nS[(height-j-1-1)*width + i];
    }
}

void bounce(){
    int cellnum;

    for(int i =0; i<width-2 ; i++){ 
        for(int j =0; j<height-2 ; j++){

            cellnum = i + j*width;
            if (bar[cellnum]){

                float _nN  = nN[cellnum];
                float _nS  = nS[cellnum];
                float _nE  = nE[cellnum];
                float _nW  = nW[cellnum];
                float _nNE = nNE[cellnum];
                float _nNW = nNW[cellnum];
                float _nSE = nSE[cellnum];
                float _nSW = nSW[cellnum];

                nN[cellnum - width]                             =       _nS ;            nS[cellnum] = 0;
                nNW[cellnum - width - 1]                        =       _nSE;           nSE[cellnum] = 0;
                nW[cellnum - 1]                                 =       _nE ;            nE[cellnum] = 0;
                nS[cellnum + width]                             =       _nN ;            nN[cellnum] = 0;
                nSW[cellnum + width - 1]                        =       _nNE;           nNE[cellnum] = 0;
                nE[cellnum + 1]                                 =       _nW ;            nW[cellnum] = 0;
                nNE[cellnum - width + 1]                        =       _nSW;           nSW[cellnum] = 0;
                nSE[cellnum + width + 1]                        =       _nNW;           nNW[cellnum] = 0;

            }

        }
    }

}


void collide(){

    for(int i =1; i<width-1 ; i++){
        for(int j =1; j<height-1 ; j++){

            float x = i + j*width;
            if (bar[x] == 0){
                rho[x] = n0[x] + nN[x] + nE[x] + nS[x] + nW[x] +
                	   nNE[x] + nSE[x] + nSW[x] + nNW[x] ;
                
                if(rho[x] > 0){
                    ux[x] = (nE[x] - nW[x] + nNE[x] + nSE[x] - nSW[x] - nNW[x]) * 1/rho[x]  ;
                    uy[x] = (nN[x] - nS[x] + nNE[x] - nSE[x] - nSW[x] + nNW[x]) * 1/rho[x]  ;

                    //Pre-compute some convenient constants
                    float one9th_rho = one9th* rho[x];
                    float one36th_rho = one36th * rho[x];
                    float vx3 = 3 * ux[x];
                    float vy3 = 3 * uy[x];
                    float vx2 = ux[x] * ux[x];
                    float vy2 = uy[x] * uy[x];
                    float vxvy2 = 2 * ux[x] * uy[x];
                    float v2 = vx2 + vy2;
                    float speed2 = v2;
                    float v215 = 1.5 * v2;
                    
                    //Update densities
                    nE[x]  += omega * (   one9th_rho * (1 + vx3       + 4.5*vx2        - v215) - nE[x]);
                    nW[x]  += omega * (   one9th_rho * (1 - vx3       + 4.5*vx2        - v215) - nW[x]);
                    nN[x]  += omega * (   one9th_rho * (1 + vy3       + 4.5*vy2        - v215) - nN[x]);
                    nS[x]  += omega * (   one9th_rho * (1 - vy3       + 4.5*vy2        - v215) - nS[x]);
                    nNE[x] += omega * (  one36th_rho * (1 + vx3 + vy3 + 4.5*(v2+vxvy2) - v215) - nNE[x]);
                    nNW[x] += omega * (  one36th_rho * (1 - vx3 + vy3 + 4.5*(v2-vxvy2) - v215) - nNW[x]);
                    nSE[x] += omega * (  one36th_rho * (1 + vx3 - vy3 + 4.5*(v2-vxvy2) - v215) - nSE[x]);
                    nSW[x] += omega * (  one36th_rho * (1 - vx3 - vy3 + 4.5*(v2+vxvy2) - v215) - nSW[x]);
                    
                    //Conserve mass
                    n0[x]   = rho[x] - (nE[x]+nW[x]+nN[x]+nS[x]+nNE[x]+nSE[x]+nNW[x]+nSW[x]);
                }
            }

        }

    }
}


//--------------- GRAPHICS ---------------
void speedToColor(float speed, float maxSpeed, float& r, float& g, float& b){   
    float t = speed/maxSpeed;
    t = fmin(fmax(t, 0.0f), 1.0f);
    r = fmin(2.0f * t, 1.0f);
    g = fmin(2.0f * t, 2.0f - 2.0f * t);
    b = fmax(1.0f - 2.0f * t, 0.0f);
}


int main(){



    initialize(25, 16, 10, u0);

    std::vector<unsigned char> pixels(width * height * 3);

    if (!glfwInit()) {
            std::cerr << "Erreur GLFW" << std::endl;
            return -1;
        }

        GLFWwindow* window = glfwCreateWindow(width, height, "LBM", nullptr, nullptr);
        if (!window) {
            std::cerr << "Erreur fenetre" << std::endl;
            glfwTerminate();
            return -1;
        }

    glfwMakeContextCurrent(window);


    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    while (!glfwWindowShouldClose(window)) {


        // Physics computation
        collide();
        stream();
        bounce();


        for (int i = 0; i < width * height; i++) {

            if (bar[i]) {
                pixels[i*3] = 0; pixels[i*3+1] = 0; pixels[i*3+2] = 0;
                continue;
            }
            
            float speed = sqrt(ux[i]*ux[i] + uy[i]*uy[i]);
            float r, g, b;

            speedToColor(speed, u0, r, g, b); 
            pixels[i*3]   = (unsigned char)(r * 255);
            pixels[i*3+1] = (unsigned char)(g * 255);
            pixels[i*3+2] = (unsigned char)(b * 255);
        }

        //update texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

        // drawing
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            glTexCoord2f(0,0); glVertex2f(-1,-1);
            glTexCoord2f(1,0); glVertex2f( 1,-1);
            glTexCoord2f(1,1); glVertex2f( 1, 1);
            glTexCoord2f(0,1); glVertex2f(-1, 1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}