#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

class Display {

    private:
        GLFWwindow* window_; 

        int width_, height_;

        float* speed_host_;
        float* bar_host_;


    public:

        GLuint texture_;

        Display(int w, int h);

        void render(float* speed, float* bar);
        bool isOpen();

        
        ~Display();

};