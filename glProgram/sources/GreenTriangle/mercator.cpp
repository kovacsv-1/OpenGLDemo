//=============================================================================================
// Zöld háromszög: A framework.h osztályait felhasználó megoldás
//=============================================================================================
#include "framework.h"

// Texture vertex shader
const char* textureVertSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoord;

    out vec2 vTexCoord;

    void main() {
        gl_Position = vec4(position, 0, 1);
        vTexCoord = texCoord;
    }
)";

// Texture fragment shader
const char* textureFragSource = R"(
#version 330
precision highp float;

uniform sampler2D texture1;  // The map texture
uniform float timeOfDay;     // Current time in hours (0-24)
in vec2 vTexCoord;           // Texture coordinates
out vec4 fragmentColor;

void main() {
    // Get the color from the map texture
    vec4 mapColor = texture(texture1, vTexCoord);

    // Calculate longitude from the texture x-coordinate (vTexCoord.x is the longitude)
    float longitude = vTexCoord.x * 360.0 - 180.0;  // Convert [0, 1] range to [-180, 180] longitude range
    
    // Calculate the sun's position based on time of day
    float napx = (360 - timeOfDay * 15.0f) / 180;
    float napy = 23.0f / 170.0f;  // Sun's latitude at Tropic of Cancer (23°N)
    napx -= 1;

    // Map texture coordinates to spherical coordinates
    vec2 napPos = vec2(napx, napy);
    vec2 vTexCoordNormalized = vec2(vTexCoord.x * 2.0 - 1.0, vTexCoord.y * 2.0 - 1.0);

    // Convert from normalized texture coordinates to lat/lon (spherical coordinates)
    float napLon = napPos.x * 180;
    float napLat = degrees(atan(sinh(napPos.y * log(tan(3.141592 / 4.0f + radians(85.0f) / 2.0f)))));
    float vlon = vTexCoordNormalized.x * 180;
    float vlat = degrees(atan(sinh(vTexCoordNormalized.y * log(tan(3.141592 / 4.0f + radians(85.0f) / 2.0f)))));

    // Convert lat/lon to radians for spherical calculations
    float napLonRad = radians(napLon);
    float napLatRad = radians(napLat);
    vec3 napSp = vec3(cos(napLatRad) * cos(napLonRad),
                      sin(napLatRad),
                      cos(napLatRad) * sin(napLonRad));
    
    float vlonRad = radians(vlon);
    float vlatRad = radians(vlat);
    vec3 vSp = vec3(cos(vlatRad) * cos(vlonRad),
                    sin(vlatRad),
                    cos(vlatRad) * sin(vlonRad));

    // Dot product between sun and fragment positions
    float dotProduct = clamp(dot(napSp, vSp), -1.0f, 1.0f);
    float angle = acos(dotProduct);

    // Apply 50% black shading for night regions, else use the original map color
    if (angle > 3.141592 / 2.0) {  // If the angle is greater than 90 degrees, it's night
        fragmentColor = mix(mapColor, vec4(0.0, 0.0, 0.0, 1.0), 0.5);  // 50% black for night
    } else {
        fragmentColor = mapColor;  // Daytime: no change
    }
}
)";

// Regular vertex shader
const char* vertSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec3 position;

    void main() {
        gl_Position = vec4(position.xy, 0, 1);
    }
)";

// Regular fragment shader
const char* fragSource = R"(
    #version 330
    precision highp float;

    uniform vec3 color;
    out vec4 fragmentColor;

    void main() {
        fragmentColor = vec4(color, 1);
    }
)";

float clamp(float x, float minVal, float maxVal) {
	if (x < minVal) return minVal;
	if (x > maxVal) return maxVal;
	return x;
}

float radians(float degrees) {
	return degrees * (M_PI / 180.0f);
}

float degrees(float radians) {
	return radians * (180.0f / M_PI);
}

vec2 getLonLat(vec2 point) {
	float lon = point.x * 180;
	float lat = degrees(atan(sinh(point.y * log(tan(3.141592 / 4.0f + radians(85.0f) / 2.0f)))));
	return vec2(lon, lat);
}

vec3 getSpherePos(vec2 point) {

	vec2 lonlat = getLonLat(point);
	float lonRad = radians(lonlat.x);
	float latRad = radians(lonlat.y);

	return {
		cos(latRad) * cos(lonRad),
		sin(latRad),
		cos(latRad) * sin(lonRad)
	};
}

float bezartszog(vec3 p0, vec3 p1) {
	return acos(clamp(dot(p0, p1), -1.0f, 1.0f));
}

float bezartszog(vec2 p0, vec2 p1) {
	vec3 s1 = getSpherePos(p0);
	vec3 s2 = getSpherePos(p1);
	return bezartszog(s1, s2);
}

class Terkep {
	unsigned int textureID;
	Texture* texture;
	unsigned int VAO, VBO[2];
	std::vector<vec2> vtx = { vec2(-1.0f, -1.0f), vec2(1.0f, -1.0f), vec2(1.0f, 1.0f), vec2(-1.0f, 1.0f) };
	std::vector<unsigned char> encodedBytes = {
		252, 252, 252, 252, 252, 252, 252, 252, 252, 0, 9, 80, 1, 148, 13, 72, 13, 140, 25, 60, 21, 132, 41, 12, 1, 28, 25, 128, 61, 0, 17, 4, 29, 124, 81, 8, 37, 116,
		89, 0, 69, 16, 5, 48, 97, 0, 77, 0, 25, 8, 1, 8, 253, 253, 253, 253, 101, 10, 237, 14, 237, 14, 241, 10, 141, 2, 93, 14, 121, 2, 5, 6, 93, 14, 49, 6, 57, 26, 89,
		18, 41, 10, 57, 26, 89, 18, 41, 14, 1, 2, 45, 26, 89, 26, 33, 18, 57, 14, 93, 26, 33, 18, 57, 10, 93, 18, 5, 2, 33, 18, 41, 2, 5, 2, 5, 6, 89, 22, 29, 2, 1, 22,
		37, 2, 1, 6, 1, 2, 97, 22, 29, 38, 45, 2, 97, 10, 1, 2, 37, 42, 17, 2, 13, 2, 5, 2, 89, 10, 49, 46, 25, 10, 101, 2, 5, 6, 37, 50, 9, 30, 89, 10, 9, 2, 37, 50, 5,
		38, 81, 26, 45, 22, 17, 54, 77, 30, 41, 22, 17, 58, 1, 2, 61, 38, 65, 2, 9, 58, 69, 46, 37, 6, 1, 10, 9, 62, 65, 38, 5, 2, 33, 102, 57, 54, 33, 102, 57, 30, 1, 14,
		33, 2, 9, 86, 9, 2, 21, 6, 13, 26, 5, 6, 53, 94, 29, 26, 1, 22, 29, 0, 29, 98, 5, 14, 9, 46, 1, 2, 5, 6, 5, 2, 0, 13, 0, 13, 118, 1, 2, 1, 42, 1, 4, 5, 6, 5, 2, 4,
		33, 78, 1, 6, 1, 6, 1, 10, 5, 34, 1, 20, 2, 9, 2, 12, 25, 14, 5, 30, 1, 54, 13, 6, 9, 2, 1, 32, 13, 8, 37, 2, 13, 2, 1, 70, 49, 28, 13, 16, 53, 2, 1, 46, 1, 2, 1,
		2, 53, 28, 17, 16, 57, 14, 1, 18, 1, 14, 1, 2, 57, 24, 13, 20, 57, 0, 2, 1, 2, 17, 0, 17, 2, 61, 0, 5, 16, 1, 28, 25, 0, 41, 2, 117, 56, 25, 0, 33, 2, 1, 2, 117,
		52, 201, 48, 77, 0, 121, 40, 1, 0, 205, 8, 1, 0, 1, 12, 213, 4, 13, 12, 253, 253, 253, 141 };

public:
	~Terkep() {
		delete texture;
	}

	void initTexture() {
		decodeTexture();
	}

	void decodeTexture() {
		std::vector<unsigned char> pixels;

		// Dekódolási logika
		for (unsigned char byte : encodedBytes) {
			int H = byte >> 2;
			int I = byte & 0x03;
			for (int i = 0; i < H + 1; ++i) {
				pixels.push_back(I);
				if (pixels.size() >= 4096) break;
			}
			if (pixels.size() >= 4096) break;
		}

		// Színkonverzió
		vec4* rgb = new vec4[64 * 64];
		for (int i = 0; i < 64 * 64; ++i) {
			switch (pixels[i]) {
			case 0: rgb[i] = { 1.0f, 1.0f, 1.0f, 1.0f }; break;
			case 1: rgb[i] = { 0.0f, 0.0f, 1.0f, 1.0f }; break;
			case 2: rgb[i] = { 0.0f, 1.0f, 0.0f, 1.0f }; break;
			case 3: rgb[i] = { 0.0f, 0.0f, 0.0f, 1.0f }; break;
			}
		}
		texture = new Texture(64, 64);
		UploadTexture(rgb);
	}

	void UploadTexture(vec4* image) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_FLOAT, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
		glGenBuffers(2, &VBO[0]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec3), &vtx[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

		std::vector<vec2> uvs = { vec2(0.0f, 0.0f), vec2(1.0f, 0.0f), vec2(1.0f, 1.0f), vec2(0.0f, 1.0f) };

		glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(vec3), &uvs[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	}

	void Draw() {
		texture->Bind(0);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLE_FAN, 0, vtx.size());
	}
};

//---------------------------
class Allomas {
	//---------------------------
	unsigned int vao, vbo;	// GPU
protected:
	std::vector<vec3> vtx;	// CPU
public:
	Allomas() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		int nf = (int)(sizeof(vec3) / sizeof(float)) >= 4 ? 4 : (int)(sizeof(vec3) / sizeof(float));
		glVertexAttribPointer(0, nf, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	std::vector<vec3>& Vtx() { return vtx; }

	void updateGPU() {	// CPU -> GPU
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec3), &vtx[0], GL_DYNAMIC_DRAW);
	}
	void Bind() { glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); } // aktiválás
	void Draw(GPUProgram* prog, int type, vec3 color) {
		if (vtx.size() > 0) {
			prog->setUniform(color, "color");
			updateGPU();
			Bind();
			glDrawArrays(type, 0, (int)vtx.size());
		}
	}

	vec2 getLonLat() {
		float lon = vtx.at(0).x * 180;
		float lat = degrees(atan(sinh(vtx.at(0).y * log(tan(3.141592 / 4.0f + radians(85.0f) / 2.0f)))));
		return vec2(lon, lat);
	}

	vec3 getSpherePos() {

		vec2 lonlat = getLonLat();
		float lonRad = radians(lonlat.x);
		float latRad = radians(lonlat.y);

		return {
			cos(latRad) * cos(lonRad),
			sin(latRad),
			cos(latRad) * sin(lonRad)
		};
	}

	virtual ~Allomas() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class Ut {
private:
	unsigned int vao, vbo;	// GPU
protected:
	std::vector<vec3> vtx;	// CPU
public:
	Ut(Allomas* p0, Allomas* p1) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		int nf = (int)(sizeof(vec3) / sizeof(float)) >= 4 ? 4 : (int)(sizeof(vec3) / sizeof(float));
		glVertexAttribPointer(0, nf, GL_FLOAT, GL_FALSE, 0, NULL);
		vec3 s1 = p0->getSpherePos();
		vec3 s2 = p1->getSpherePos();

		float angle = acos(clamp(dot(s1, s2), -1.0f, 1.0f));
		printf("Distance: %.2f km\n", angle * 40000.0 / (2 * M_PI));

		// 100 pontos interpoláció
		for (float t = 0; t <= 1; t += 0.01) {
			vec3 p = slerp(s1, s2, t);

			// Gömbi -> Mercator
			float lat = degrees(asin(p.y));
			float lon = degrees(atan2(p.z, p.x));

			float x = lon / 180.0f;

			const float maxLat = 85.0f;
			float maxMercatorY = log(tan(3.141592 / 4.0f + radians(maxLat) / 2.0f));

			float latRad = radians(clamp(lat, -maxLat, maxLat));
			float mercatorY = log(tan(3.141592 / 4.0f + latRad / 2.0f));
			float y = mercatorY / maxMercatorY;

			vtx.push_back({ x, y, 1 });
		}
	}

	vec3 slerp(vec3 p0, vec3 p1, float t) {
		float theta = acos(clamp(dot(p0, p1), -1.0f, 1.0f));
		return (sin(theta * (1 - t)) * p0 + sin(theta * t) * p1) / sin(theta);
	}


	std::vector<vec3>& Vtx() { return vtx; }

	void updateGPU() {	// CPU -> GPU
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(vec3), &vtx[0], GL_DYNAMIC_DRAW);
	}
	void Bind() { glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); } // aktiválás
	void Draw(GPUProgram* prog, int type, vec3 color) {
		if (vtx.size() > 0) {
			prog->setUniform(color, "color");
			updateGPU();
			Bind();
			glDrawArrays(type, 0, (int)vtx.size());
		}
	}
	virtual ~Ut() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//saját cucc vége

const int winWidth = 600, winHeight = 600;

class MercatorApp : public glApp {
	GPUProgram* gpuProgram;	   // csúcspont és pixel árnyalók
	GPUProgram* textureProgram;  // For map
	int ora = 0;
	std::vector<Allomas*> points = std::vector<Allomas*>();
	std::vector<Ut*> lines = std::vector<Ut*>();
	Terkep terkep;
public:
	MercatorApp() : glApp("Terkep") {}

	void onKeyboard(int key) {
		switch (key) {
		case 'n':
			ora = (ora + 1) % 24;
			refreshScreen();
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
			Allomas* point = new Allomas();
			point->Vtx() = { vec3(A, B, 1) };
			if (!points.empty()) {
				if (!lines.empty()) lines.pop_back();
				Ut* line0 = new Ut(points.back(), point);
				lines.push_back(line0);
				Ut* line1 = new Ut(points.front(), point);
				lines.push_back(line1);
			}
			points.push_back(point);
			for (int i = 0; i < lines.size(); ++i) {
				lines.at(i)->Draw(gpuProgram, GL_LINE_STRIP, vec3(1.0f, 1.0f, 0.0f));
			}
			for (int i = 0; i < points.size(); ++i) {
				points.at(i)->Draw(gpuProgram, GL_POINTS, vec3(1.0f, 0.0f, 0.0f));
			}
			refreshScreen();
		}
	}

	// Inicializáció, 
	void onInitialization() {
		glPointSize(10);
		glLineWidth(3);
		gpuProgram = new GPUProgram(vertSource, fragSource);
		textureProgram = new GPUProgram(textureVertSource, textureFragSource);
		terkep.decodeTexture();
	}

	// Ablak újrarajzolás
	void onDisplay() {
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, winWidth, winHeight);


		// Draw map first with texture shader
		textureProgram->Use();

		// Pass the time of day to the shader
		textureProgram->setUniform((float)ora, "timeOfDay");
		// Draw map first with texture shader
		terkep.Draw();

		// Switch to color shader for other elements
		gpuProgram->Use();

		for (auto line : lines) line->Draw(gpuProgram, GL_LINE_STRIP, vec3(1, 1, 0));
		for (auto point : points) point->Draw(gpuProgram, GL_POINTS, vec3(1, 0, 0));
	}
};

MercatorApp app;

