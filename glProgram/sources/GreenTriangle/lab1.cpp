#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector>
#include <cmath>

struct Point {
    float x, y;
};

class Object {
protected:
    GLuint VAO, VBO;
    std::vector<Point> vertices;

public:
    Object() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
    }

    void updateGPU() {
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Point), vertices.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Point), (void*)0);
        glEnableVertexAttribArray(0);
    }

    virtual void draw(GLenum primitiveType) {
        glBindVertexArray(VAO);
        glDrawArrays(primitiveType, 0, vertices.size());
    }
};

class PointCollection : public Object {
public:
    void addPoint(float x, float y) {
        vertices.push_back({ x, y });
        updateGPU();
    }

    void drawPoints() {
        draw(GL_POINTS);
    }
};

class Line {
    Point p1, p2;
    float A, B, C;
public:
    Line(Point _p1, Point _p2) : p1(_p1), p2(_p2) {
        A = p2.y - p1.y;
        B = p1.x - p2.x;
        C = A * p1.x + B * p1.y;
    }
};

class LineCollection : public Object {
public:
    void addLine(Point p1, Point p2) {
        vertices.push_back(p1);
        vertices.push_back(p2);
        updateGPU();
    }

    void drawLines() {
        draw(GL_LINES);
    }
};

void onDisplay() {
    glClear(GL_COLOR_BUFFER_BIT);
    // draw objects here
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int x, int y) {
    if (key == 'p') {
        // handle point input mode
    }
    else if (key == 'l') {
        // handle line input mode
    }
}

void onMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // Convert screen coords to normalized device coords
    }
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("OpenGL Geometry");
    glewInit();

    glutDisplayFunc(onDisplay);
    glutKeyboardFunc(onKeyboard);
    glutMouseFunc(onMouse);

    glutMainLoop();
    return 0;
}