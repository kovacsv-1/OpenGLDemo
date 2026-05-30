#include "framework.h"

const int windowWidth = 600, windowHeight = 600;

vec3 operator/(const vec3& v, const vec3& f) {
	return vec3(v.x / f.x, v.y / f.y, v.z / f.z);
}

struct Material {
	vec3 n = vec3(0.17f, 0.35f, 1.5f);
	vec3 kappa = vec3(3.1f, 2.7f, 1.9f);
	vec3 ka, kd, ks;
	float  shininess;
	bool rough = true, reflective = false, refractive = false, seeThrough = false;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* (float)M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
	vec3 shade(vec3 N, vec3 V, vec3 L, vec3 inRad) {
		float cosTheta = dot(N, L);// unit vecs
		if (cosTheta < 0) return vec3(0, 0, 0);  // self shadow
		vec3 diffuseRad = inRad * kd * cosTheta; // diffuse
		vec3 H = normalize(V + L);
		float cosDelta = dot(N, H);
		if (cosDelta < 0) return diffuseRad;
		return diffuseRad + inRad * ks * pow(cosDelta, shininess);
	}
	vec3 reflect(vec3 V, vec3 N) {
		return V - N * dot(N, V) * 2;
	};
	vec3 refract(vec3 V, vec3 N, float eta) {
		float c = -dot(V, N);
		float disc = 1 - eta * eta * (1 - c * c);
		if (disc < 0) return vec3(0, 0, 0);
		return eta * V + (eta * c - sqrt(disc)) * N;
	}
	vec3 Fresnel(vec3 V, vec3 N) {
		float cosa = -dot(V, N);
		vec3 one(1, 1, 1);
		vec3 F0 = ((n - one) * (n - one) + kappa * kappa) /
			((n + one) * (n + one) + kappa * kappa);
		return F0 + (one - F0) * pow(1 - cosa, 5);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	bool out;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); out = true; }
	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class Sphere : public Intersectable {
	vec3 center;
	float radius;
public:
	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center; radius = _radius; material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		return hit;
	}
};

class Cylinder : public Intersectable {
	vec3 baseCenter;    // Base point of cylinder
	vec3 axis;          // Normalized axis vector (points along cylinder)
	float radius;
	float height;

public:
	Cylinder(const vec3& _baseCenter, const vec3& _axis, float _radius, float _height, Material* _material) {
		baseCenter = _baseCenter;
		axis = normalize(_axis);
		radius = _radius;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;

		vec3 d = ray.dir;
		vec3 delta = ray.start - baseCenter;
		vec3 a = axis;

		vec3 d_proj = d - a * dot(d, a);
		vec3 delta_proj = delta - a * dot(delta, a);

		float A = dot(d_proj, d_proj);
		float B = 2 * dot(d_proj, delta_proj);
		float C = dot(delta_proj, delta_proj) - radius * radius;

		float discr = B * B - 4 * A * C;
		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-B - sqrt_discr) / (2 * A);
		float t2 = (-B + sqrt_discr) / (2 * A);

		for (float t : {t1, t2}) {
			if (t <= 0) continue;
			vec3 p = ray.start + ray.dir * t;
			float proj = dot(p - baseCenter, a);
			if (proj >= 0 && proj <= height) {
				vec3 radial = p - baseCenter - a * proj;
				if (dot(ray.dir, radial) > 0 && !ray.out) continue; // inside surface, skip
				hit.t = t;
				hit.position = p;
				hit.normal = normalize(radial);
				hit.material = material;
				break;
			}
		}
		return hit;
	}
};

class Cone : public Intersectable {
	vec3 apex;
	vec3 axis;
	float height;
	float angle;  // In radians

public:
	Cone(const vec3& _apex, const vec3& _axis, float _angle, float _height, Material* _material) {
		apex = _apex;
		axis = normalize(_axis);
		angle = _angle;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;

		vec3 d = ray.dir;
		vec3 delta = ray.start - apex;
		vec3 a = axis;

		float cos2 = cos(angle) * cos(angle);
		float sin2 = sin(angle) * sin(angle);

		float D_a = dot(d, a);
		float delta_a = dot(delta, a);

		vec3 d_perp = d - a * D_a;
		vec3 delta_perp = delta - a * delta_a;

		float A = cos2 * dot(d_perp, d_perp) - sin2 * D_a * D_a;
		float B = 2 * (cos2 * dot(d_perp, delta_perp) - sin2 * D_a * delta_a);
		float C = cos2 * dot(delta_perp, delta_perp) - sin2 * delta_a * delta_a;

		float discr = B * B - 4 * A * C;
		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-B - sqrt_discr) / (2 * A);
		float t2 = (-B + sqrt_discr) / (2 * A);

		for (float t : {t1, t2}) {
			if (t <= 0) continue;
			vec3 p = ray.start + ray.dir * t;
			float h = dot(p - apex, a);
			if (h >= 0 && h <= height) {
				hit.t = t;
				hit.position = p;

				vec3 axisOffset = a * h;
				vec3 normal = normalize(p - apex - axisOffset * (1 + tan(angle) * tan(angle)));
				hit.normal = normal;
				hit.material = material;
				break;
			}
		}
		return hit;
	}
};

class CheckerBoardPlane : public Intersectable {
	vec3 origin;       // Center of the plane
	vec3 normal;       // Should be (0, 1, 0) for a horizontal plane
	float size;        // Half-size of the plane (i.e., 10 for 20x20)
	Material* material1; // Blue tile
	Material* material2; // White tile
public:
	CheckerBoardPlane(vec3 _origin, vec3 _normal, float _size, Material* m2, Material* m1)
		: origin(_origin), normal(normalize(_normal)), size(_size), material1(m1), material2(m2) {
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;

		float denom = dot(ray.dir, normal);
		if (fabs(denom) < 1e-6f) return hit; // parallel

		float t = dot(origin - ray.start, normal) / denom;
		if (t <= 0) return hit;

		vec3 hitPoint = ray.start + ray.dir * t;

		// Project hit point to checkerboard coordinates (assume axes aligned with X and Z)
		float x = hitPoint.x + size;
		float z = hitPoint.z + size;

		// Check if hit is inside the 20x20 square
		if (x < 0 || x >= 2 * size || z < 0 || z >= 2 * size) return hit;

		int ix = int(floor(x));
		int iz = int(floor(z));

		// Determine if tile is blue or white
		bool isWhite = ((ix + iz) % 2 == 1); // (0.5, 0.5, -1) is white
		hit.t = t;
		hit.position = hitPoint;
		hit.normal = normal;
		hit.material = isWhite ? material2 : material1;

		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * (float)windowSize * (float)windowWidth / (float)windowHeight;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir, true);
	}

	void Animate(float state) {

		vec3 eye = vec3(4 * sin(state * M_PI / 4), 1, 4 * cos(state * M_PI / 4)), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		set(eye, lookat, vup, fov);

		//vec3 d = eye - lookat;
		//eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		//set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float Epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		//vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		vec3 eye = vec3(0, 1, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		Material* Gold = new Material(vec3(0.8f, 0.6f, 0.1f) * 0.05, vec3(1, 1, 1), 0); // kd is visual only
		Gold->rough = false;
		Gold->reflective = true;
		Gold->refractive = false;
		Gold->seeThrough = false;
		Gold->n = vec3(0.17f, 0.35f, 1.5f);       // refractive index for RGB
		Gold->kappa = vec3(3.1f, 2.7f, 1.9f);     // extinction coefficient

		Material* Water = new Material(vec3(0.0f), vec3(1.0f), 100); // no diffuse, full specular
		Water->rough = false;
		Water->reflective = false;
		Water->refractive = true;
		Water->seeThrough = false;
		Water->n = vec3(1.3f);
		Water->kappa = vec3(0.0f);
		Water->ka = vec3(0.0f);

		// Yellow plastic cylinder
		Material* YellowPlastic = new Material(vec3(0.3f, 0.2f, 0.1f), vec3(2.0f), 50);
		YellowPlastic->rough = true;

		// Cyan cone
		Material* CyanCone = new Material(vec3(0.1f, 0.2f, 0.3f), vec3(2.0f), 100);
		CyanCone->rough = true;

		// Magenta cone
		Material* MagentaCone = new Material(vec3(0.3f, 0.0f, 0.2f), vec3(2.0f), 20);
		MagentaCone->rough = true;

		// Checkerboard tile materials
		Material* BlueTile = new Material(vec3(0.0f, 0.1f, 0.3f), vec3(0.0f), 0);
		BlueTile->rough = true;

		Material* WhiteTile = new Material(vec3(0.3f, 0.3f, 0.3f), vec3(0.0f), 0);
		WhiteTile->rough = true;

		// Origin at (0, -1, 0), facing up, 20x20 checkerboard
		objects.push_back(new CheckerBoardPlane(vec3(0, -1, 0), vec3(0, 1, 0), 10.0f, BlueTile, WhiteTile));

		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1f), 0.3f, 2.0f, YellowPlastic));
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1f, -1, -0.05f), 0.2f, 2.0f, CyanCone));

		objects.push_back(new Cylinder(vec3(1.0f, -1.0f, 0.0f), vec3(0.1f, 1.0f, 0.0f), 0.3f, 2.0f, Gold));
		objects.push_back(new Cylinder(vec3(0.0f, -1.0f, -0.8f), vec3(-0.2f, 1.0f, -0.1f), 0.3f, 2.0f, Water));
		objects.push_back(new Cone(vec3(0.0f, 1.0f, 0.8f), vec3(0.2f, -1.0f, 0.0f), 0.2f, 2.0f, MagentaCone));
	}

	void render(std::vector<vec3>& image) {
		//float timeStart = getElapsedTime();
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec3(color.x, color.y, color.z);
			}
		}
		printf("Rendered\n");
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = -bestHit.normal;
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		if (depth > 5) return La; // max recursion depths

		vec3 outRadiance = hit.material->ka * La;
		if (hit.material->rough) {
			for (Light* light : lights) { //DirectLight
				Ray shadowRay(hit.position + hit.normal * Epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->reflective) {
			Ray reflectedRay(hit.position + hit.normal * Epsilon, hit.material->reflect(ray.dir, hit.normal), ray.out);
			outRadiance += trace(reflectedRay, depth + 1) * hit.material->Fresnel(ray.dir, hit.normal);
		}
		if (hit.material->refractive) {
			float eta = (ray.out) ? (1.0f / hit.material->n.x) : hit.material->n.x;
			vec3 refractedDir = hit.material->refract(ray.dir, hit.normal, eta);
			if (length(refractedDir) > 0) {
				vec3 offset = (ray.out ? -hit.normal : hit.normal) * Epsilon;
				vec3 N = (ray.out) ? -hit.normal : hit.normal;
				vec3 F = hit.material->Fresnel(ray.dir, N);
				Ray refractedRay(hit.position + offset, refractedDir, !ray.out);
				outRadiance += trace(refractedRay, depth + 1) * (vec3(1, 1, 1) - F);
			}
		}

		return outRadiance;
	}

	void Animate(int state) {
		camera.Animate(state);
	}
};

Scene scene;
GPUProgram gpuProgram; // vertex and fragment shaders

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() { fragmentColor = texture(textureUnit, texcoord); }
)";

class FullScreenTexturedQuad : public Geometry<vec2> {
	unsigned int textureID = 0;
	int width, height;

public:
	FullScreenTexturedQuad(int w, int h) : width(w), height(h) {
		// Initialize texture once
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		vtx = { vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1) };
		updateGPU();
	}

	void LoadTexture(std::vector<vec3>& image) {
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, image.data());

		if (GLenum err = glGetError(); err != GL_NO_ERROR) {
			printf("Texture update error: %x\n", err);
		}
	}

	void Bind(int textureUnit) {
		glActiveTexture(GL_TEXTURE0 + textureUnit);
		glBindTexture(GL_TEXTURE_2D, textureID);
	}

	void Draw() {
		Geometry<vec2>::Bind();
		Bind(0);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}

	~FullScreenTexturedQuad() {
		if (textureID != 0) glDeleteTextures(1, &textureID);
	}
};


class RaytraceApp : public glApp {
	Geometry<vec2>* triangle;  // geometria
	GPUProgram gpuProgram;	   // csúcspont és pixel árnyalók
	FullScreenTexturedQuad* fullScreenTexturedQuad;
	int campos = 0;
public:
	RaytraceApp() : glApp("Ray tracing") {}

	// Inicializáció, 
	void onInitialization() {
		glViewport(0, 0, windowWidth, windowHeight);
		scene.build();
		fullScreenTexturedQuad = new FullScreenTexturedQuad(600, 600);
		gpuProgram.create(vertexSource, fragmentSource); 	// create program for the GPU
	}

	// Ablak újrarajzolás
	void onDisplay() {
		std::vector<vec3> image(windowWidth * windowHeight);
		scene.render(image); 						// Execute ray casting
		fullScreenTexturedQuad->LoadTexture(image); // copy image to GPU as a texture
		fullScreenTexturedQuad->Draw();				// Display rendered image on screen
	}
	/*void onTimeElapsed(float startTime, float endTime) {
		scene.Animate(endTime - startTime);
		refreshScreen();
	}*/

	void onKeyboard(int key) {
		if (key == 'a') {
			campos = (campos + 1) % 8;
			scene.Animate(campos);
			refreshScreen();
		}
	}
} app;