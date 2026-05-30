//=============================================================================================
// Zöld háromszög: A framework.h osztályait felhasználó megoldás
//=============================================================================================
#include "framework.h"

// csúcspont árnyaló
const char* vertSource = R"(
	#version 330				
    precision highp float;

	layout(location = 0) in vec2 cP;	// 0. bemeneti regiszter

	void main() {
		gl_Position = vec4(cP.x, cP.y, 0, 1); 	// bemenet már normalizált eszközkoordinátákban
	}
)";

// pixel árnyaló
const char* fragSource = R"(
	#version 330
    precision highp float;

	uniform vec3 color;			// konstans szín
	out vec4 fragmentColor;		// pixel szín

	void main() {
		fragmentColor = vec4(color, 1); // RGB -> RGBA
	}
)";

//---------------------------
template<class T>
class Object {
	//---------------------------
	unsigned int vao, vbo;	// GPU
protected:
	std::vector<T> vtx;	// CPU
public:
	Object() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		int nf = (int)(sizeof(T) / sizeof(float)) >= 4 ? 4 : (int)(sizeof(T) / sizeof(float));
		glVertexAttribPointer(0, nf, GL_FLOAT, GL_FALSE, 0, NULL);
	}
	std::vector<T>& Vtx() { return vtx; }

	void updateGPU() {	// CPU -> GPU
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(T), &vtx[0], GL_DYNAMIC_DRAW);
	}
	void Bind() { glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); } // aktiválás
	void Draw(GPUProgram* prog, int type, vec3 color) {
		if (vtx.size() > 0) {
			prog->setUniform(color, "color");
			Bind();
			glDrawArrays(type, 0, (int)vtx.size());
		}
	}
	virtual ~Object() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class PointCollection {
private:
	std::vector<Object<vec3>*> points;
public:
	void addPoint(Object<vec3>* point) {
		points.push_back(point);
		printf("Point %0.1f, %0.1f added\n", point->Vtx().at(0).x, point->Vtx().at(0).y);
	}

	void updateGPU() { for (auto point : points) point->updateGPU(); }

	void Draw(GPUProgram* prog, int type, vec3 color) {
		for (auto point : points) point->Draw(prog, type, color);
	}

	Object<vec3>* selectPoint(vec2 mousePos) {
		for (auto point : points)
			if (((mousePos.x - point->Vtx().at(0).x) * (mousePos.x - point->Vtx().at(0).x) + (mousePos.y - point->Vtx().at(0).y) * (mousePos.y - point->Vtx().at(0).y)) < 0.03)
				return point;
		return NULL;
	}
};

class Line : public Object<vec3> {
private:
	float A, B, C;
public:
	Line() {};

	Line(Object<vec3>* p0, Object<vec3>* p1) {
		vec2 o1 = vec2(p0->Vtx().at(0).x, p0->Vtx().at(0).y);
		vec2 o2 = vec2(p1->Vtx().at(0).x, p1->Vtx().at(0).y);
		if (p0->Vtx().at(0).x > p1->Vtx().at(0).x) {
			o2 = vec2(p0->Vtx().at(0).x, p0->Vtx().at(0).y);
			o1 = vec2(p1->Vtx().at(0).x, p1->Vtx().at(0).y);
		}
		printf("Line added\n\tImplicit: %0.1f x + %0.1f y + %0.1f = 0\n\tParametric: r(t) = (%0.1f, %0.1f) + (%0.1f, %0.1f)t\n",
			o2.y - o1.y, -(o2.x - o1.x), o2.x * o1.y - o1.x * o2.y,
			o1.x, o1.y, o2.x - o1.x, o2.y - o1.y);
		A = o2.y - o1.y; B = -(o2.x - o1.x); C = o2.x * o1.y - o1.x * o2.y;
		if ((A + C) / (-B) > (B + C) / (-A)) { //itt normalizálom
			o1 = vec2((B + C) / (-A), 1);
			o2 = vec2((-B + C) / (-A), -1);
		}
		else {
			o1 = vec2(1, (A + C) / (-B));
			o2 = vec2(-1, (-A + C) / (-B));
		}
		vtx.push_back(vec3(o1.x, o1.y, 0.0f));
		vtx.push_back(vec3(o2.x, o2.y, 0.0f));
	}

	Object<vec3>* intersection(Line* line) {
		float A1 = line->Vtx().at(1).y - line->Vtx().at(0).y;
		float B1 = -(line->Vtx().at(1).x - line->Vtx().at(0).x);
		float C1 = line->Vtx().at(1).x * line->Vtx().at(0).y - line->Vtx().at(0).x * line->Vtx().at(1).y;
		float D = A * B1 - B * A1;
		if (D == 0) return NULL;
		Object<vec3>* ret = new Object();
		ret->Vtx() = { vec3(((-C) * B1 + C1 * B) / D, (A * (-C1) + A1 * C) / D, 1) };
		return ret;
	}

	bool close(Object<vec3>* point) {
		return 0.01 > abs(A * point->Vtx().at(0).x + B * point->Vtx().at(0).y + C) / sqrt(A * A + B * B);
	}

	std::vector<vec3> normalized() { return Vtx(); }

	void push(Object<vec3>* point) {
		C = -(A * point->Vtx().at(0).x + B * point->Vtx().at(0).y);
		//printf("%d\n", C);
		vec2 o1, o2;
		if ((A + C) / (-B) > (B + C) / (-A)) { //itt is normalizálom
			o1 = vec2((B + C) / (-A), 1);
			o2 = vec2((-B + C) / (-A), -1);
		}
		else {
			o1 = vec2(1, (A + C) / (-B));
			o2 = vec2(-1, (-A + C) / (-B));
		}
		vtx.at(0) = vec3(o1.x, o1.y, 0.0f);
		vtx.at(1) = vec3(o2.x, o2.y, 0.0f);
	}
};

class LineCollection {
private:
	std::vector<Line*> lines;
public:
	LineCollection() {}

	void addLine(Line* line) {
		lines.push_back(line);
	}

	void updateGPU() { for (auto line : lines) line->updateGPU(); }

	void Draw(GPUProgram* prog, int type, vec3 color) {
		for (auto line : lines) line->Draw(prog, type, color);
	}

	Line* selectLine(vec2 mousePos) {
		Object<vec3> point = Object<vec3>();
		point.Vtx() = { vec3(mousePos.x, mousePos.y, 1) };
		for (auto line : lines) if (line->close(&point)) return line;
		return NULL;
	}
};

//saját cucc vége

const int winWidth = 600, winHeight = 600;

class PointsAndLinesApp : public glApp {
	GPUProgram* gpuProgram;	   // csúcspont és pixel árnyalók
	PointCollection* points;
	LineCollection* lines;
	int mode = 0;
	Object<vec3>* selectedPoint = NULL;
	Line* selectedLine = NULL;
	bool moving = false;
public:
	PointsAndLinesApp() : glApp("Pontok es egyenesek") {}

	void onKeyboard(int key) {
		switch (key) {
		case 'p':
			selectedPoint = NULL;
			selectedLine = NULL;
			printf("Define points\n");
			mode = 0;
			break;
		case 'l':
			selectedPoint = NULL;
			selectedLine = NULL;
			printf("Define lines\n");
			mode = 1;
			break;
		case 'm':
			selectedPoint = NULL;
			selectedLine = NULL;
			printf("Move\n");
			mode = 2;
			break;
		case 'i':
			selectedPoint = NULL;
			selectedLine = NULL;
			printf("Intersect\n");
			mode = 3;
			break;
		default:
			break;
		}
	}

	void onMousePressed(MouseButton button, int pX, int pY) {
		if (button == MOUSE_LEFT) {
			float A, B;
			A = 2.0f * pX / 600.0f - 1.0f;
			B = 1.0f - 2.0f * pY / 600.0f;
			vec2 mousePos = vec2(A, B);
			Object<vec3>* point = NULL;
			Line* line = NULL;
			switch (mode) {
			case 0:
				point = new Object<vec3>();
				point->Vtx() = { vec3(mousePos.x, mousePos.y, 1.0f) };
				points->addPoint(point);
				break;
			case 1:
				point = points->selectPoint(mousePos);
				if (selectedPoint == NULL) selectedPoint = point;
				else if (point != NULL && selectedPoint != point) {
					lines->addLine(new Line(selectedPoint, point));
					selectedPoint = NULL;
				}
				break;
			case 2:
				moving = true;
				selectedLine = lines->selectLine(mousePos);
				break;
			case 3:
				line = lines->selectLine(mousePos);
				if (selectedLine == NULL) selectedLine = line;
				else if (line != NULL && selectedLine != line) {
					Object<vec3>* intersected = selectedLine->intersection(line);
					if (intersected != NULL) points->addPoint(intersected);
					selectedLine = NULL;
				}
				break;
			default:
				break;
			}
			lines->updateGPU();
			points->updateGPU();
			lines->Draw(gpuProgram, GL_LINES, vec3(0.0f, 1.0f, 1.0f));
			points->Draw(gpuProgram, GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
			refreshScreen();
		}
	}

	void onMouseReleased(MouseButton button, int pX, int pY) {
		if (button == MOUSE_LEFT && mode == 2) {
			moving = false;
			selectedLine = NULL;
			lines->updateGPU();
			points->updateGPU();
			lines->Draw(gpuProgram, GL_LINES, vec3(0.0f, 1.0f, 1.0f));
			points->Draw(gpuProgram, GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
			refreshScreen();
		}
	}

	void onMouseMotion(int pX, int pY) {
		if (moving && mode == 2 && selectedLine != NULL) {
			Object<vec3>* follow = new Object<vec3>();
			float A, B;
			A = 2.0f * pX / 600.0f - 1.0f;
			B = 1.0f - 2.0f * pY / 600.0f;
			follow->Vtx() = { vec3(A, B, 1) };
			selectedLine->push(follow);
			lines->updateGPU();
			points->updateGPU();
			lines->Draw(gpuProgram, GL_LINES, vec3(0.0f, 1.0f, 1.0f));
			points->Draw(gpuProgram, GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
			refreshScreen();
		}
	}

	// Inicializáció, 
	void onInitialization() {
		glPointSize(10);
		glLineWidth(3);
		gpuProgram = new GPUProgram(vertSource, fragSource);
		points = new PointCollection;
		lines = new LineCollection;
	}

	// Ablak újrarajzolás
	void onDisplay() {
		glClearColor(0, 0, 0, 0);     // háttér szín
		glClear(GL_COLOR_BUFFER_BIT); // rasztertár törlés
		glViewport(0, 0, winWidth, winHeight);
		lines->Draw(gpuProgram, GL_LINES, vec3(0.0f, 1.0f, 1.0f));
		points->Draw(gpuProgram, GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
	}
};

PointsAndLinesApp app;

