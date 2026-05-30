//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"

class Texture {
	//---------------------------
	unsigned int textureId = 0;
public:
#ifdef FILE_OPERATIONS
	Texture(const fs::path pathname, bool transparent = false, int sampling = GL_LINEAR) {
		if (textureId == 0) glGenTextures(1, &textureId);  				// azonosító generálás
		glBindTexture(GL_TEXTURE_2D, textureId);    // kötés
		unsigned int width, height;
		unsigned char* pixels;
		if (transparent) {
			lodepng_decode32_file(&pixels, &width, &height, pathname.string().c_str());
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					float sum = 0;
					for (int c = 0; c < 3; ++c) {
						sum += pixels[4 * (x + y * width) + c];
					}
					pixels[4 * (x + y * width) + 3] = sum / 6;
				}
			}

			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels); // GPU-ra
		}
		else {
			lodepng_decode24_file(&pixels, &width, &height, pathname.string().c_str());
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels); // GPU-ra
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampling); // szűrés
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampling);
		printf("%s, w: %d, h: %d\n", pathname.string().c_str(), width, height);
	}
#endif
	Texture(int width, int height) {
		glGenTextures(1, &textureId); // azonosító generálása
		glBindTexture(GL_TEXTURE_2D, textureId);    // ez az aktív innentől
		// procedurális textúra előállítása programmal
		const vec3 yellow(1, 1, 0), blue(0, 0, 1);
		std::vector<vec3> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	Texture(int width, int height, std::vector<vec3>& image) {
		glGenTextures(1, &textureId); // azonosító generálása
		glBindTexture(GL_TEXTURE_2D, textureId);    // ez az aktív innentől
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void Bind(int textureUnit) {
		glActiveTexture(GL_TEXTURE0 + textureUnit); // aktiválás
		glBindTexture(GL_TEXTURE_2D, textureId); // piros nyíl
	}
	~Texture() {
		if (textureId > 0) glDeleteTextures(1, &textureId);
	}
};

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
	//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

vec3 getRotationAxis(vec3 from, vec3 to) {
	return normalize(cross(from, to));
}

float getRotationAngle(vec3 from, vec3 to) {
	return acos(dot(normalize(from), normalize(to)));
}


typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

const int windowWidth = 600, windowHeight = 600;

//---------------------------
struct Camera { // 3D camera
	//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 45.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 50;
	}
	mat4 V() { return lookAt(wEye, wLookat, wVup); }
	mat4 P() { return perspective(fov, asp, fp, bp); }
	void Animate(int campos) {
		vec3 eye = vec3(4 * sin(campos * M_PI / 4), 1, 4 * cos(campos * M_PI / 4)), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		set(eye, lookat, vup, fov);
	}

	void set(vec3 eye, vec3 lookat, vec3 vup, float fov) {
		wEye = eye;
		wLookat = lookat;
		wVup = vup;
		this->fov = fov;
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};
//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
		out vec3 worldPos;

		void main() {
			gl_Position = MVP * vec4(vtxPos, 1.0); // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			worldPos = wPos.xyz;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform vec3 triangleP0[256];  // max 256 háromszög
		uniform vec3 triangleP1[256];
		uniform vec3 triangleP2[256];
		uniform int numTriangles;
		uniform vec3 lightDir; // normalized

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
		uniform bool useTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		in vec3 worldPos;
		
        out vec4 fragmentColor; // output goes to frame buffer

		bool rayIntersectsTriangle(vec3 orig, vec3 dir, vec3 v0, vec3 v1, vec3 v2) {
			const float EPSILON = 1e-4;
			vec3 edge1 = v1 - v0;
			vec3 edge2 = v2 - v0;
			vec3 h = cross(dir, edge2);
			float a = dot(edge1, h);
			if (abs(a) < EPSILON) return false;
			float f = 1.0 / a;
			vec3 s = orig - v0;
			float u = f * dot(s, h);
			if (u < 0.0 || u > 1.0) return false;
			vec3 q = cross(s, edge1);
			float v = f * dot(dir, q);
			if (v < 0.0 || u + v > 1.0) return false;
			float t = f * dot(edge2, q);
			return t > EPSILON;
		}

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;

			vec3 texColor = useTexture ? texture(diffuseTexture, texcoord).rgb : vec3(1.0);
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);

				// árnyékteszt: sugár a fény irányába, kis offset a felszínről
				bool inShadow = false;
				//vec3 origin = vec3(gl_FragCoord.xyz); // alternatíva: wPos, ha elérhető
				vec3 origin = worldPos + N * 0.01;
				for (int t = 0; t < numTriangles; t++) {
					if (rayIntersectsTriangle(origin, -lightDir, triangleP0[t], triangleP1[t], triangleP2[t])) {
						inShadow = true;
						break;
					}
				}

				if (inShadow) {
					float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
					radiance += ka * lights[i].La+
								(kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].La * 2; // csak ambient
				} else {
					float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
					radiance += ka * lights[i].La +
								(kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
				}
			}

			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource/*, "fragmentColor"*/); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		int textureUnit = 0;
		setUniform(textureUnit, "diffuseTexture");
		setUniform(state.texture != nullptr, "useTexture");
		if (state.texture != nullptr) {
			state.texture->Bind(textureUnit);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Object3D {
protected:
	GLuint vao = 0, vbo = 0;
	int vertexCount = 0;
	std::vector<VertexData> vertices;
public:
	Object3D() {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
	}

	const std::vector<VertexData>& getVertices() const { return vertices; }

	virtual ~Object3D() {
		if (vbo) glDeleteBuffers(1, &vbo);
		if (vao) glDeleteVertexArrays(1, &vao);
	}

	void uploadVertexData(const std::vector<VertexData>& vertices) {
		vertexCount = vertices.size();
		this->vertices = vertices;
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(VertexData), vertices.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0); // position
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));

		glEnableVertexAttribArray(1); // normal
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));

		glEnableVertexAttribArray(2); // texcoord
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	virtual void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vertexCount);
	}
};

class Quad : public Object3D {
public:
	// Default: 2x2 négyzet az xz síkban
	Quad(vec3 center = vec3(0, 0, 0), vec3 normal = vec3(0, 1, 0), vec2 size = vec2(2, 2)) {
		std::vector<VertexData> verts;

		// Két tengely a síkban, ami a normálra merőleges
		vec3 tangent = normalize(cross(normal, vec3(0, 0, 1)));
		if (length(tangent) < 1e-6f) tangent = normalize(cross(normal, vec3(0, 1, 0)));
		vec3 bitangent = normalize(cross(normal, tangent));

		vec3 halfU = tangent * (size.x / 2.0f);
		vec3 halfV = bitangent * (size.y / 2.0f);

		vec3 p0 = center - halfU - halfV;
		vec3 p1 = center + halfU - halfV;
		vec3 p2 = center + halfU + halfV;
		vec3 p3 = center - halfU + halfV;

		vec3 n = normalize(normal); // biztos, ami biztos

		verts.push_back({ p0, n, vec2(0, 0) });
		verts.push_back({ p1, n, vec2(1, 0) });
		verts.push_back({ p2, n, vec2(1, 1) });

		verts.push_back({ p0, n, vec2(0, 0) });
		verts.push_back({ p2, n, vec2(1, 1) });
		verts.push_back({ p3, n, vec2(0, 1) });

		uploadVertexData(verts);
	}
};


class Cylinder : public Object3D {
public:
	Cylinder(vec3 baseCenter = vec3(0, 0, 0), vec3 axis = vec3(0, 1, 0), float height = 2.0f, float radius = 1.0f) {
		const int segments = 6;
		std::vector<VertexData> verts;

		vec3 w = normalize(axis);
		vec3 topCenter = baseCenter + w * height;

		// Orthonormal basis
		vec3 u = normalize(cross(w, vec3(1, 0, 0)));
		if (length(u) < 1e-6f) u = normalize(cross(w, vec3(0, 1, 0)));
		vec3 v = normalize(cross(w, u));

		for (int i = 0; i < segments; ++i) {
			float a0 = (float)(i + 0.5f) / segments * 2.0f * M_PI;
			float a1 = (float)(i + 1.5f) / segments * 2.0f * M_PI;

			vec3 dir0 = cos(a0) * u + sin(a0) * v;
			vec3 dir1 = cos(a1) * u + sin(a1) * v;

			vec3 p0 = baseCenter + radius * dir0;
			vec3 p1 = baseCenter + radius * dir1;
			vec3 p2 = p0 + w * height;
			vec3 p3 = p1 + w * height;

			vec2 uv0((float)i / segments, 0);
			vec2 uv1((float)(i + 1) / segments, 0);
			vec2 uv2((float)i / segments, 1);
			vec2 uv3((float)(i + 1) / segments, 1);

			// Sima normálok a henger oldalára
			vec3 n0 = dir0;
			vec3 n1 = dir1;

			// Oldal
			verts.push_back({ p0, n0, uv0 });
			verts.push_back({ p1, n1, uv1 });
			verts.push_back({ p2, n0, uv2 });

			verts.push_back({ p1, n1, uv1 });
			verts.push_back({ p3, n1, uv3 });
			verts.push_back({ p2, n0, uv2 });
			/*
			// Alap (alsó körlap)
			vec3 baseNormal = -w;
			verts.push_back({ baseCenter, baseNormal, vec2(0.5f, 0.5f) });
			verts.push_back({ p1, baseNormal, vec2(0.5f + 0.5f * cos(a1), 0.5f + 0.5f * sin(a1)) });
			verts.push_back({ p0, baseNormal, vec2(0.5f + 0.5f * cos(a0), 0.5f + 0.5f * sin(a0)) });

			// Tető (felső körlap)
			vec3 topP0 = p0 + w * height;
			vec3 topP1 = p1 + w * height;
			vec3 topNormal = w;
			verts.push_back({ topCenter, topNormal, vec2(0.5f, 0.5f) });
			verts.push_back({ topP0, topNormal, vec2(0.5f + 0.5f * cos(a0), 0.5f + 0.5f * sin(a0)) });
			verts.push_back({ topP1, topNormal, vec2(0.5f + 0.5f * cos(a1), 0.5f + 0.5f * sin(a1)) });
			*/
		}

		uploadVertexData(verts);
	}
};


class Cone : public Object3D {
public:
	Cone(vec3 apex, vec3 axis, float height, float angle) {
		const int segments = 12; // növelt felbontás a simább megjelenéshez
		std::vector<VertexData> verts;

		vec3 w = normalize(axis);
		vec3 baseCenter = apex + w * height;
		float radius = tan(angle) * height;

		// Orthonormal basis for circular base
		vec3 u = normalize(cross(w, vec3(1, 0, 0)));
		if (length(u) < 1e-6f) u = normalize(cross(w, vec3(0, 1, 0)));
		vec3 v = normalize(cross(w, u));

		for (int i = 0; i < segments; ++i) {
			float a0 = (float)(i + 0.5f) / segments * 2.0f * M_PI;
			float a1 = (float)(i + 1.5f) / segments * 2.0f * M_PI;

			vec3 dir0 = cos(a0) * u + sin(a0) * v;
			vec3 dir1 = cos(a1) * u + sin(a1) * v;

			vec3 p0 = baseCenter + radius * dir0;
			vec3 p1 = baseCenter + radius * dir1;

			// Vertex normálok: a felülethez húzott irány (apex -> körív)
			vec3 n_apex = normalize(cross(p0 - apex, p1 - apex));  // átlag a két irányból
			vec3 n0 = normalize(cross(cross(p0 - apex, w), p0 - apex));
			vec3 n1 = normalize(cross(cross(p1 - apex, w), p1 - apex));

			// Oldallap
			verts.push_back({ apex, n_apex, vec2(0.5f, 1.0f) });
			verts.push_back({ p0, n0, vec2((float)i / segments, 0.0f) });
			verts.push_back({ p1, n1, vec2((float)(i + 1) / segments, 0.0f) });
			/*
			// Alaplap
			vec3 baseNormal = -w;
			verts.push_back({ baseCenter, baseNormal, vec2(0.5f, 0.5f) });
			verts.push_back({ p1, baseNormal, vec2(0.5f + 0.5f * cos(a1), 0.5f + 0.5f * sin(a1)) });
			verts.push_back({ p0, baseNormal, vec2(0.5f + 0.5f * cos(a0), 0.5f + 0.5f * sin(a0)) });
			*/
		}

		uploadVertexData(verts);
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Object3D* geometry3D;
	vec3 scaling, translation, rotationAxis;
	float rotationAngle;
public:

	Object(Shader* _shader, Material* _material, Texture* _texture, Object3D* _geometry3D) :
		scaling(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry3D = _geometry3D;
	}

	Object(Shader* _shader, Material* _material, Object3D* _geometry3D) :
		scaling(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = nullptr;
		material = _material;
		geometry3D = _geometry3D;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = translate(translation) * rotate(rotationAngle, rotationAxis) * scale(scaling);
		Minv = scale(vec3(1 / scaling.x, 1 / scaling.y, 1 / scaling.z)) * rotate(-rotationAngle, rotationAxis) * translate(-translation);
	}

	vec3 transformPoint(vec3 localPoint) const {
		mat4 M, Minv;
		const_cast<Object*>(this)->SetModelingTransform(M, Minv);
		vec4 world = M * vec4(localPoint.x, localPoint.y, localPoint.z, 1.0f);
		return vec3(world.x, world.y, world.z);
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		//state.MVP = state.M * state.V * state.P;
		state.MVP = state.P * state.V * state.M; // projection matrix is first
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		if (geometry3D) geometry3D->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	std::vector<vec3> triP0, triP1, triP2;
	Camera camera; // 3D camera
	std::vector<Light> lights;
public:

	void Build() {
		// Shader
		Shader* phongShader = new PhongShader();

		int texWidth = 20;
		int texHeight = 20;

		std::vector<vec3> pixels;
		pixels.reserve(texWidth * texHeight);

		vec3 blue(0.0f, 0.2f, 0.6f);
		vec3 white(0.6f, 0.6f, 0.6f);

		for (int y = 0; y < texHeight; ++y) {
			for (int x = 0; x < texWidth; ++x) {
				bool isEven = (x + y) % 2 == 0;
				pixels.push_back(isEven ? blue : white);
			}
		}

		Texture* checkerTexture = new Texture(texWidth, texHeight, pixels);

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.3, 0.2, 0.1);
		material0->ks = vec3(2, 2, 2);
		material0->ka = material0->kd * vec3(3);
		material0->shininess = 50;

		Object* yellowCylinder = new Object(phongShader, material0, new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 2, 0.3));
		objects.push_back(yellowCylinder);

		Material* cyanMaterial = new Material;
		cyanMaterial->kd = vec3(0.1f, 0.2f, 0.3f);
		cyanMaterial->ks = vec3(2.0f);
		cyanMaterial->ka = cyanMaterial->kd * 3.0f;
		cyanMaterial->shininess = 100.0f;

		Object* cyanCone = new Object(phongShader, cyanMaterial, nullptr,
			new Cone(vec3(0, 1, 0), vec3(-0.1f, -1, -0.05f), 2.0f, 0.2f));
		objects.push_back(cyanCone);

		Material* magentaMaterial = new Material;
		magentaMaterial->kd = vec3(0.3f, 0.0f, 0.2f);
		magentaMaterial->ks = vec3(2.0f);
		magentaMaterial->ka = magentaMaterial->kd * 3.0f;
		magentaMaterial->shininess = 20.0f;

		Object* magentaCone = new Object(phongShader, magentaMaterial, nullptr,
			new Cone(vec3(0, 1, 0.8f), vec3(0.2f, -1, 0), 2.0f, 0.2f));
		objects.push_back(magentaCone);

		Material* goldMaterial = new Material;
		goldMaterial->kd = vec3(0.17f, 0.35f, 1.5f);
		goldMaterial->ks = vec3(3.1f, 2.7f, 1.9f);
		goldMaterial->ka = goldMaterial->kd * 3.0f;
		goldMaterial->shininess = 150.0f;

		Object* goldCylinder = new Object(phongShader, goldMaterial, nullptr,
			new Cylinder(vec3(1, -1, 0), vec3(0.1f, 1, 0), 2.0f, 0.3f));
		objects.push_back(goldCylinder);

		Material* waterMaterial = new Material;
		waterMaterial->kd = vec3(1.3f);
		waterMaterial->ks = vec3(0.1f);
		waterMaterial->ka = waterMaterial->kd * 3.0f;
		waterMaterial->shininess = 80.0f;

		Object* waterCylinder = new Object(phongShader, waterMaterial, nullptr,
			new Cylinder(vec3(0, -1, -0.8f), vec3(-0.2f, 1, -0.1f), 2.0f, 0.3f));
		objects.push_back(waterCylinder);

		Material* neutralMaterial = new Material;
		neutralMaterial->kd = vec3(1.0f);
		neutralMaterial->ks = vec3(0.0f);
		neutralMaterial->ka = vec3(0.0f);
		neutralMaterial->shininess = 100.0f;

		Object* checkerQuad = new Object(phongShader, neutralMaterial, checkerTexture,
			new Quad(vec3(0, -1, 0), vec3(0, 1, 0), vec2(20, 20)));
		objects.push_back(checkerQuad);

		// Camera
		camera.wEye = vec3(0, 1, 4);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		//Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(1, 1, 1, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.4, 0.4, 0.4);
		lights[0].Le = vec3(2, 2, 2);

		triP0.clear(); triP1.clear(); triP2.clear();
		for (Object* obj : objects) {
			Object3D* mesh = obj->geometry3D;
			if (!mesh) continue;
			const std::vector<VertexData>& verts = mesh->getVertices();
			for (size_t i = 0; i + 2 < verts.size(); i += 3) {
				triP0.push_back(obj->transformPoint(verts[i + 0].position));
				triP1.push_back(obj->transformPoint(verts[i + 1].position));
				triP2.push_back(obj->transformPoint(verts[i + 2].position));
			}
		}

		std::vector<float> triP0Flat, triP1Flat, triP2Flat;

		for (size_t i = 0; i < triP0.size(); ++i) {
			triP0Flat.push_back(triP0[i].x);
			triP0Flat.push_back(triP0[i].y);
			triP0Flat.push_back(triP0[i].z);

			triP1Flat.push_back(triP1[i].x);
			triP1Flat.push_back(triP1[i].y);
			triP1Flat.push_back(triP1[i].z);

			triP2Flat.push_back(triP2[i].x);
			triP2Flat.push_back(triP2[i].y);
			triP2Flat.push_back(triP2[i].z);
		}


		GLint currentProgram;
		glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);

		GLint loc0 = glGetUniformLocation(currentProgram, "triangleP0");
		GLint loc1 = glGetUniformLocation(currentProgram, "triangleP1");
		GLint loc2 = glGetUniformLocation(currentProgram, "triangleP2");
		GLint locN = glGetUniformLocation(currentProgram, "numTriangles");

		vec3 dir = normalize(vec3(-lights[0].wLightPos.x, -lights[0].wLightPos.y, -lights[0].wLightPos.z));
		glUniform3fv(glGetUniformLocation(currentProgram, "lightDir"), 1, &dir.x);

		glUniform1i(locN, (GLint)triP0.size()); // number of triangles
		glUniform3fv(loc0, triP0.size(), triP0Flat.data());
		glUniform3fv(loc1, triP1.size(), triP1Flat.data());
		glUniform3fv(loc2, triP2.size(), triP2Flat.data());

	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(int campos) {
		camera.Animate(campos);
	}
};

class EngineApp : public glApp {
	Scene scene;
	int campos = 0;
	int camangle = 0;
public:
	EngineApp() : glApp("3D Engine-ke") {}

	void onInitialization() {
		glViewport(0, 0, windowWidth, windowHeight);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		scene.Build();
	}
	void onDisplay() {
		glClearColor(0.3f, 0.3f, 1.0f, 1.0f);				// background color 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
		scene.Render();
	}

	void onKeyboard(int key) {
		if (key == 'a') {
			campos = (campos + 1) % 8;
			scene.Animate(campos);
			refreshScreen();
		}
	}
} app;