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
#include "textures.h"

float radians(int degrees) {
	return degrees * (float)M_PI / 180.0f;
}

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

const int windowWidth = 640, windowHeight = 360;

//---------------------------
struct Camera { // 3D camera
	//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 90.0f * (float)M_PI / 180.0f;
		fp = 0.2f; bp = 50;
	}
	mat4 V() { return lookAt(wEye, wLookat, wVup); }
	mat4 P() { return perspective(fov, asp, fp, bp); }
	void Animate(vec3 pos, vec3 facing) {
		//vec3 eye = vec3(4 * sin(campos * M_PI / 4), 1, 4 * cos(campos * M_PI / 4)), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 90 * (float)M_PI / 180;
		set(pos, facing, vec3(0, 1, 0), fov);
	}

	void set(vec3 eye, vec3 lookat, vec3 vup, float fov) {
		wEye = eye;
		wLookat = lookat;
		//wVup = vup;
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

uniform mat4 MVP, M; //, Minv;
uniform vec3 wEye;

layout(location = 0) in vec3 vtxPos;
layout(location = 1) in vec3 vtxNorm;
layout(location = 2) in vec2 vtxUV;

out vec3 wNormal;
out vec3 wView;
out vec2 texcoord;
out vec3 worldPos;

void main() {
    vec4 wPos = M * vec4(vtxPos, 1.0);
    worldPos = wPos.xyz;

    wView = wEye - worldPos;

    // Correct normal transformation — use mat3 if Minv is correct
    wNormal = normalize(transpose(inverse(mat3(M))) * vtxNorm);

    texcoord = vtxUV;
    gl_Position = MVP * vec4(vtxPos, 1.0);
}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
precision highp float;

struct Light {
    vec3 La;
    vec3 Le;
    vec4 wLightPos; // w = 0: directional, w = 1: point light
};

struct Material {
    vec3 kd, ks, ka;
    float shininess;
};

uniform vec3 triangleP0[256];
uniform vec3 triangleP1[256];
uniform vec3 triangleP2[256];
uniform int numTriangles;

uniform Material material;
uniform Light lights[8];
uniform int nLights;
uniform sampler2D diffuseTexture;
uniform bool useTexture;
uniform vec3 channelsEnabled;

in vec3 wNormal;
in vec3 wView;
in vec2 texcoord;
in vec3 worldPos;

out vec4 fragmentColor;

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

    vec4 texColor = useTexture ? texture(diffuseTexture, texcoord) : vec4(1.0);

    if (texColor.a < 0.01)
        discard;

    vec3 ka = material.ka * texColor.rgb;
    vec3 kd = material.kd * texColor.rgb;

    vec3 radiance = vec3(0);

    for (int i = 0; i < nLights; i++) {
        vec3 L, H;
        bool inShadow = false;
        float lightDistance;

        if (lights[i].wLightPos.w == 1.0) {
            // Point light
            vec3 lightPos = lights[i].wLightPos.xyz;
            L = normalize(lightPos - worldPos);
            lightDistance = distance(lightPos, worldPos);
            H = normalize(L + V);

            vec3 origin = worldPos + N * 0.01;

            for (int t = 0; t < numTriangles; t++) {
                if (rayIntersectsTriangle(origin, L, triangleP0[t], triangleP1[t], triangleP2[t])) {
                    vec3 edge1 = triangleP1[t] - triangleP0[t];
                    vec3 edge2 = triangleP2[t] - triangleP0[t];
                    vec3 h = cross(L, edge2);
                    float a = dot(edge1, h);
                    float f = 1.0 / a;
                    vec3 s = origin - triangleP0[t];
                    float u = f * dot(s, h);
                    vec3 q = cross(s, edge1);
                    float v = f * dot(L, q);
                    float t_hit = f * dot(edge2, q);
                    if (t_hit > 0.0 && t_hit < lightDistance) {
                        inShadow = true;
                        break;
                    }
                }
            }

            float cost = max(dot(N, L), 0.0);
            float cosd = max(dot(N, H), 0.0);

            if (inShadow) {
                radiance += ka * lights[i].La +
                            (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].La * 0.5;
            } else {
                float attenuation = 1.0 / (1.0 + 0.1 * lightDistance + 0.01 * lightDistance * lightDistance);
                radiance += ka * lights[i].La +
                            (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le * attenuation;
            }
        } else {
            // Directional light
            L = normalize(lights[i].wLightPos.xyz);
            H = normalize(L + V);

            vec3 origin = worldPos + N * 0.01;

            for (int t = 0; t < numTriangles; t++) {
                if (rayIntersectsTriangle(origin, L, triangleP0[t], triangleP1[t], triangleP2[t])) {
                    inShadow = true;
                    break;
                }
            }

            float cost = max(dot(N, L), 0.0);
            float cosd = max(dot(N, H), 0.0);

            if (inShadow) {
                radiance += ka * lights[i].La +
                            (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].La * 0.3;
            } else {
                radiance += ka * lights[i].La +
                            (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
            }
        }
    }
    fragmentColor = vec4(radiance * channelsEnabled, texColor.a);
}
)";
public:
	PhongShader() { create(vertexSource, fragmentSource/*, "fragmentColor"*/); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		//setUniform(state.Minv, "Minv");
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
			setUniform(state.lights[i].La, std::string("lights[") + std::to_string(i) + "].La");
			setUniform(state.lights[i].Le, std::string("lights[") + std::to_string(i) + "].Le");
			setUniform(state.lights[i].wLightPos, std::string("lights[") + std::to_string(i) + "].wLightPos");
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
	Quad(vec3 center = vec3(0, 0, 0), vec3 normal = vec3(0, 1, 0), vec2 size = vec2(2, 2), vec2 tiling = vec2(1, 1)) {
		std::vector<VertexData> verts;

		// Compute tangent vectors orthogonal to the normal
		vec3 crossResult = cross(normal, vec3(0, 0, 1));
		if (length(crossResult) < 1e-6f) {
			crossResult = cross(normal, vec3(0, 1, 0));
		}
		vec3 tangent = normalize(crossResult);
		vec3 bitangent = normalize(cross(normal, tangent));

		// Scale the axis vectors by half the size
		vec3 halfU = tangent * (size.x / 2.0f);
		vec3 halfV = bitangent * (size.y / 2.0f);

		// Calculate the four corners of the quad
		vec3 p0 = center - halfU - halfV;
		vec3 p1 = center + halfU - halfV;
		vec3 p2 = center + halfU + halfV;
		vec3 p3 = center - halfU + halfV;

		// Ensure normal is unit length
		vec3 n = normalize(normal);

		// Define vertices for two triangles with proper UVs
		verts.push_back({ p0, n, vec2(0, 0) });
		verts.push_back({ p1, n, vec2(tiling.x, 0) });
		verts.push_back({ p2, n, vec2(tiling.x, tiling.y) });

		verts.push_back({ p0, n, vec2(0, 0) });
		verts.push_back({ p2, n, vec2(tiling.x, tiling.y) });
		verts.push_back({ p3, n, vec2(0, tiling.y) });

		uploadVertexData(verts);
	}
};


class Cylinder : public Object3D {
public:
	Cylinder(vec3 baseCenter = vec3(0, 0, 0), vec3 axis = vec3(0, 1, 0), float height = 2.0f, float radius = 1.0f, int segments = 6, vec2 tiling = vec2(1, 1)) {
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
			verts.push_back({ p0, n0, uv0 * tiling });
			verts.push_back({ p1, n1, uv1 * tiling });
			verts.push_back({ p2, n0, uv2 * tiling });

			verts.push_back({ p1, n1, uv1 * tiling });
			verts.push_back({ p3, n1, uv3 * tiling });
			verts.push_back({ p2, n0, uv2 * tiling });
			/*
			// Alap (alsó körlap)
			vec3 baseNormal = -w;
			verts.push_back({ baseCenter, baseNormal, vec2(0.5f, 0.5f) });
			verts.push_back({ p1, baseNormal, vec2(0.5f + 0.5f * cos(a1), 0.5f + 0.5f * sin(a1)) });
			verts.push_back({ p0, baseNormal, vec2(0.5f + 0.5f * cos(a0), 0.5f + 0.5f * sin(a0)) });

			// Tetõ (felsõ körlap)
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
	Cone(vec3 apex, vec3 axis, float height, float angle, int segments = 12) {
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

class Dodecahedron : public Object3D {
public:
	Dodecahedron(vec3 center = vec3(0), float scale = 1.0f) {
		std::vector<VertexData> verts;

		const float phi = (1.0f + sqrt(5.0f)) * 0.5f; // golden ratio
		const float a = 1.0f / sqrt(3.0f);
		const float b = a / phi;
		const float c = a * phi;

		// 20 vertices of a dodecahedron
		std::vector<vec3> positions = {
			vec3(a,  a,  a), vec3(a,  a, -a), vec3(a, -a,  a), vec3(a, -a, -a),
			vec3(-a,  a,  a), vec3(-a,  a, -a), vec3(-a, -a,  a), vec3(-a, -a, -a),
			vec3(0,  b,  c), vec3(0,  b, -c), vec3(0, -b,  c), vec3(0, -b, -c),
			vec3(b,  c,  0), vec3(b, -c,  0), vec3(-b,  c,  0), vec3(-b, -c,  0),
			vec3(c,  0,  b), vec3(c,  0, -b), vec3(-c,  0,  b), vec3(-c,  0, -b)
		};

		// Each face is a pentagon; we split it into 3 triangles (triangle fan from vertex 0 of each face)
		std::vector<std::vector<int>> faces = {
			{0, 8, 10, 2, 16},
			{0, 16, 17, 1, 12},
			{0, 12, 14, 4, 8},
			{8, 4, 18, 6, 10},
			{10, 6, 15, 13, 2},
			{2, 13, 3, 17, 16},
			{1, 17, 3, 11, 9},
			{1, 9, 5, 14, 12},
			{4, 14, 5, 19, 18},
			{6, 18, 19, 7, 15},
			{3, 13, 15, 7, 11},
			{5, 9, 11, 7, 19}
		};

		for (const auto& face : faces) {
			vec3 p0 = positions[face[0]] * scale + center;
			for (size_t i = 1; i + 1 < face.size(); ++i) {
				vec3 p1 = positions[face[i]] * scale + center;
				vec3 p2 = positions[face[i + 1]] * scale + center;

				// Compute flat normal
				vec3 n = normalize(cross(p1 - p0, p2 - p0));

				// Simple planar UVs — not suitable for detailed texturing
				verts.push_back({ p0, n, vec2(0.5f, 1.0f) });
				verts.push_back({ p1, n, vec2(0.0f, 0.0f) });
				verts.push_back({ p2, n, vec2(1.0f, 0.0f) });
			}
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
		state.MVP = state.P * state.V * state.M;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		if (geometry3D) geometry3D->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		rotationAngle = 0.8f * tend;
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	std::vector<vec3> triP0, triP1, triP2;
	std::vector<Light> lights;
	// Shader
	Shader* phongShader;
public:
	Camera camera; // 3D camera

	void Build() {

		phongShader = new PhongShader();

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

		Texture* brickTexture = new Texture(16, 16, brickWall);

		Texture* stoneBrickTexture = new Texture(16, 16, stoneBricks);

		//printf("%d", stoneBricks.size());

		Texture* glassTexture = new Texture(16, 16, glass);

		std::vector<vec3> rotatedBrick;

		for (int y = 0; y < 16; ++y) {
			for (int x = 0; x < 16; ++x) {
				vec3 color = brickWall[(x % 16) * 16 + (y % 16)];
				rotatedBrick.push_back(color);
			}
		}

		Texture* rotatedBrickTexture = new Texture(16, 16, rotatedBrick);

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

		Material* greenMaterial = new Material;
		greenMaterial->kd = vec3(0, 1, 0);
		greenMaterial->ks = vec3(2.0f);
		greenMaterial->ka = greenMaterial->kd * 3.0f;
		greenMaterial->shininess = 25.0f;

		Material* redMaterial = new Material;
		redMaterial->kd = vec3(1, 0, 0);
		redMaterial->ks = vec3(2.0f);
		redMaterial->ka = redMaterial->kd * 3.0f;
		redMaterial->shininess = 25.0f;

		Material* blueMaterial = new Material;
		blueMaterial->kd = vec3(0, 0, 1);
		blueMaterial->ks = vec3(2.0f);
		blueMaterial->ka = blueMaterial->kd * 3.0f;
		blueMaterial->shininess = 25.0f;

		Material* greyMaterial = new Material;
		greyMaterial->kd = vec3(0.1f, 0.1f, 0.1f);
		greyMaterial->ks = vec3(2.0f);
		greyMaterial->ka = greyMaterial->kd * 3.0f;
		greyMaterial->shininess = 25.0f;

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

		Object* checkerQuad = new Object(phongShader, neutralMaterial, stoneBrickTexture,
			new Quad(vec3(0, -1, 0), vec3(0, 1, 0), vec2(20, 20), vec2(20, 20)));
		objects.push_back(checkerQuad);

		Object* wall0 = new Object(phongShader, neutralMaterial, brickTexture,
			new Quad(vec3(0, 0, 10), vec3(0, 0, 1), vec2(20, 2), vec2(20, 2)));
		objects.push_back(wall0);

		Object* wall1 = new Object(phongShader, neutralMaterial, brickTexture,
			new Quad(vec3(0, 0, -10), vec3(0, 0, 1), vec2(20, 2), vec2(20, 2)));
		objects.push_back(wall1);

		Object* wall2 = new Object(phongShader, neutralMaterial, rotatedBrickTexture,
			new Quad(vec3(10, 0, 0), vec3(1, 0, 0), vec2(2, 20), vec2(2, 20)));
		objects.push_back(wall2);

		Object* wall3 = new Object(phongShader, neutralMaterial, rotatedBrickTexture,
			new Quad(vec3(-10, 0, 0), vec3(1, 0, 0), vec2(2, 20), vec2(2, 20)));
		objects.push_back(wall3);

		Object* Red = new Object(phongShader, redMaterial, nullptr,
			new Cone(vec3(0, -1, 0), vec3(0, 1, 0), 2.0f, 0.25f, 6));
		Red->translation = vec3(-9, 0, -9);
		Red->rotationAxis = vec3(0, 1, 0);
		objects.push_back(Red);

		Object* Green = new Object(phongShader, greenMaterial, nullptr,
			new Cone(vec3(0, -1, 0), vec3(0, 1, 0), 2.0f, 0.25f, 6));
		Green->translation = vec3(-7, 0, -9);
		Green->rotationAxis = vec3(0, 1, 0);
		objects.push_back(Green);

		Object* Blue = new Object(phongShader, blueMaterial, nullptr,
			new Cone(vec3(0, -1, 0), vec3(0, 1, 0), 2.0f, 0.25f, 6));
		Blue->translation = vec3(-5, 0, -9);
		Blue->rotationAxis = vec3(0, 1, 0);
		objects.push_back(Blue);

		Dodecahedron* dodecahedron = new Dodecahedron(vec3(0, 0, 0), 0.5f);
		Object* _Dodecahedron = new Object(phongShader, greyMaterial, nullptr,
			dodecahedron);
		_Dodecahedron->translation = vec3(-3, 0, -9);
		_Dodecahedron->rotationAxis = vec3(0, 1, 0);
		objects.push_back(_Dodecahedron);

		Object* magentaCylinder = new Object(phongShader, magentaMaterial, nullptr,
			new Cylinder(vec3(0, -1, 0), vec3(0, 2, 0), 2.0f, 0.5f));
		magentaCylinder->translation = vec3(-1, 0, -9);
		magentaCylinder->rotationAxis = vec3(0, 1, 0);
		objects.push_back(magentaCylinder);

		Object* yellowCylinder1 = new Object(phongShader, material0, nullptr,
			new Cylinder(vec3(0, -1, 0), vec3(0, 2, 0), 2.0f, 0.5f));
		yellowCylinder1->translation = vec3(1, 0, -9);
		yellowCylinder1->rotationAxis = vec3(0, 1, 0);
		objects.push_back(yellowCylinder1);

		Dodecahedron* magentaDodecahedron = new Dodecahedron(vec3(1.75, 0, 0), 0.25f);
		Object* MagentaDodecahedron = new Object(phongShader, magentaMaterial, nullptr,
			magentaDodecahedron);
		MagentaDodecahedron->rotationAxis = vec3(0, 1, 0);
		objects.push_back(MagentaDodecahedron);

		Dodecahedron* yellowDodecahedron = new Dodecahedron(vec3(-1.75, 0, 0), 0.25f);
		Object* YellowDodecahedron = new Object(phongShader, material0, nullptr,
			yellowDodecahedron);
		YellowDodecahedron->rotationAxis = vec3(0, 1, 0);
		objects.push_back(YellowDodecahedron);

		Object* glassQuad = new Object(phongShader, neutralMaterial, glassTexture,
			new Quad(vec3(0, 1, 0), vec3(0, -1, 0), vec2(20, 20), vec2(20, 20)));
		objects.push_back(glassQuad);

		Object* glassRoom = new Object(phongShader, neutralMaterial, glassTexture,
			new Cylinder(vec3(0, -1, 0), vec3(0, 1, 0), 2, 2.8284, 4, vec2(16, 2)));
		objects.push_back(glassRoom);

		// Camera
		camera.wEye = vec3(0, 0.5, 4);
		camera.wLookat = vec3(0, 1, 0);
		camera.wVup = vec3(0, 1, 0);

		//Lights
		lights.resize(2);
		lights[0].wLightPos = vec4(1, 0.6, 4, 1);	// ideal point -> directional light source
		lights[0].La = vec3(0.4, 0.4, 0.4);
		lights[0].Le = vec3(2, 2, 2);
		lights[1].wLightPos = vec4(0, 1, 0, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.4, 0.4, 0.4);
		lights[1].Le = vec3(1, 1, 1);

		triP0.clear(); triP1.clear(); triP2.clear();
		for (int i = 0; i < objects.size() - 2; ++i) {
			Object* obj = objects.at(i);
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

		phongShader->setUniform(vec3(1, 1, 1), "channelsEnabled");

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

	void Animate(vec3 camPos, vec3 camFacing, float tstart, float tend, vec3 rgb = vec3(1, 1, 1), int yellowEnabled = 1, int magentaEnabled = 1) {
		camera.Animate(camPos, camFacing);
		for (int i = objects.size() - 10; i < objects.size() - 2; ++i)
			objects.at(i)->Animate(tstart, tend);

		phongShader->setUniform(rgb, "channelsEnabled");

		vec3 magentaPos = vec3(0, 0, 0);
		vec3 yellowPos = vec3(0, 0, 0);

		triP0.clear(); triP1.clear(); triP2.clear();
		for (int i = 0; i < objects.size() - 4; ++i) {
			Object* obj = objects.at(i);
			Object3D* mesh = obj->geometry3D;
			if (!mesh) continue;
			const std::vector<VertexData>& verts = mesh->getVertices();
			for (size_t i = 0; i + 2 < verts.size(); i += 3) {
				triP0.push_back(obj->transformPoint(verts[i + 0].position));
				triP1.push_back(obj->transformPoint(verts[i + 1].position));
				triP2.push_back(obj->transformPoint(verts[i + 2].position));
			}
		}


		Object* obj0 = objects.at(objects.size() - 4);
		Object3D* mesh0 = obj0->geometry3D;
		const std::vector<VertexData>& verts0 = mesh0->getVertices();
		for (size_t i = 0; i < verts0.size(); ++i) {
			magentaPos += obj0->transformPoint(verts0[i].position);
		}
		magentaPos /= verts0.size();

		Object* obj1 = objects.at(objects.size() - 3);
		Object3D* mesh1 = obj1->geometry3D;
		const std::vector<VertexData>& verts1 = mesh1->getVertices();
		for (size_t i = 0; i < verts1.size(); ++i) {
			yellowPos += obj1->transformPoint(verts1[i].position);
		}
		yellowPos /= verts1.size();

		//Lights
		lights.resize(4);
		lights[0].wLightPos = vec4(yellowPos, 1);
		lights[0].La = vec3(0.4, 0.4, 0.4) * (float)magentaEnabled;
		lights[0].Le = vec3(2, 2, 0) * (float)yellowEnabled;
		lights[1].wLightPos = vec4(magentaPos, 1);
		lights[1].La = vec3(0.4, 0.4, 0.4) * (float)magentaEnabled;
		lights[1].Le = vec3(2, 0, 2) * (float)magentaEnabled;
		lights[2].wLightPos = vec4(0, 1, 0, 0);
		lights[2].La = vec3(0.4, 0.4, 0.4);
		lights[2].Le = vec3(1, 1, 1);
		lights[3].wLightPos = vec4(camera.wEye + camera.wVup * 0.1f + vec3(0.1f, 0, 0), 1);	// ideal point -> directional light source
		lights[3].La = vec3(0.4, 0.4, 0.4);
		lights[3].Le = vec3(1, 1, 0.75);

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

		glUniform1i(locN, (GLint)triP0.size()); // number of triangles
		glUniform3fv(loc0, triP0.size(), triP0Flat.data());
		glUniform3fv(loc1, triP1.size(), triP1Flat.data());
		glUniform3fv(loc2, triP2.size(), triP2Flat.data());
	}
};

class EngineApp : public glApp {
	Scene scene;
	int camAngle = 180;
	int camvelocity = 0;
	int prevx = -1, prevy = -1; // mouse position
	bool keyStates[256] = { false }; // key states
	vec3 prevpos = vec3(0, 0, 0); // previous camera position
	vec3 rgb = vec3(1, 1, 1); // channelsEnabled
	int yellowEnabled = 1; int magentaEnabled = 1;
public:
	EngineApp() : glApp("Nagyházi") {}

	void onInitialization() {
		glViewport(0, 0, windowWidth, windowHeight);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		scene.Build();
	}
	void onDisplay() {
		glClearColor(0.3f, 0.3f, 1.0f, 1.0f);				// background color 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

		scene.Render();
	}

	void onTimeElapsed(float tstart, float tend) {
		vec3 camPos = scene.camera.wEye;
		if (justendteredRadius(vec3(-9, 0, -9), 0.5)) rgb.x = abs(rgb.x - 1);
		if (justendteredRadius(vec3(-7, 0, -9), 0.5)) rgb.y = abs(rgb.y - 1);
		if (justendteredRadius(vec3(-5, 0, -9), 0.5)) rgb.z = abs(rgb.z - 1);
		if (justendteredRadius(vec3(-3, 0, -9), 0.5)) scene.camera.wVup *= -1;
		if (justendteredRadius(vec3(1, 0, -9), 0.5)) yellowEnabled = abs(yellowEnabled - 1);
		if (justendteredRadius(vec3(-1, 0, -9), 0.5)) magentaEnabled = abs(magentaEnabled - 1);
		prevpos = camPos;
		camPos.y = scene.camera.wVup.y * 0.5f;
		vec3 movement = vec3(0, 0, 0);
		float deltaTime = tend - tstart;
		if (keyStates['w']) {
			movement.x += 0.1f * sin(radians(camAngle));
			movement.z += 0.1f * cos(radians(camAngle));
		}
		if (keyStates['s']) {
			movement.x -= 0.1f * sin(radians(camAngle));
			movement.z -= 0.1f * cos(radians(camAngle));
		}
		if (keyStates['d']) {
			movement.x -= 0.1f * cos(radians(camAngle)) * scene.camera.wVup.y;
			movement.z += 0.1f * sin(radians(camAngle)) * scene.camera.wVup.y;
		}
		if (keyStates['a']) {
			movement.x += 0.1f * cos(radians(camAngle)) * scene.camera.wVup.y;
			movement.z -= 0.1f * sin(radians(camAngle)) * scene.camera.wVup.y;
		}
		if (movement != vec3(0, 0, 0))
			camPos += normalize(movement) * (deltaTime * 2);
		if (camPos.x > 9.75) camPos.x = 9.75;
		if (camPos.x < -9.75) camPos.x = -9.75;
		if (camPos.z > 9.75) camPos.z = 9.75;
		if (camPos.z < -9.75) camPos.z = -9.75;
		if (camPos.x < 2.25 && camPos.x > -2.25 && camPos.z < 2.25 && camPos.z > -2.25) {
			if (prevpos.x >= 2.25 || prevpos.x <= -2.25)
				camPos.x = (camPos.x > 0 ? 2.25 : -2.25);
			if (prevpos.z >= 2.25 || prevpos.z <= -2.25)
				camPos.z = (camPos.z > 0 ? 2.25 : -2.25);
		}
		vec3 camFacing = camPos;
		camFacing.x = camPos.x + sin(radians(camAngle));
		camFacing.z = camPos.z + cos(radians(camAngle));
		scene.Animate(camPos, camFacing, tstart, tend, rgb, yellowEnabled, magentaEnabled);
		refreshScreen();
	}

	void onKeyboard(int key) {
		keyStates[key] = true;
	}

	void onKeyboardUp(int key) {
		keyStates[key] = false;
	}

	void onMouseMotion(int x, int y) {
		if (prevx == -1) {
			prevx = x;
			prevy = y;
		}
		int dx = x - prevx;
		int dy = y - prevy;
		if (dx != 0 || dy != 0) {
			camAngle -= dx * 0.5 * scene.camera.wVup.y;
			if (camAngle < 0) camAngle += 360;
			if (camAngle >= 360) camAngle -= 360;
			vec3 camPos = scene.camera.wEye;
			/*vec3 camFacing = camPos;
			camFacing.x = camPos.x + sin(radians(camAngle));
			camFacing.z = camPos.z + cos(radians(camAngle));
			scene.Animate(camPos, camFacing);*/
		}
		prevx = x; prevy = y;
		//refreshScreen();
	}

	bool justendteredRadius(vec3 point, float radius) {
		float a0 = prevpos.x - point.x;
		float b0 = prevpos.z - point.z;
		float a1 = scene.camera.wEye.x - point.x;
		float b1 = scene.camera.wEye.z - point.z;
		return sqrt(a0 * a0 + b0 * b0) > radius && sqrt(a1 * a1 + b1 * b1) < radius;
	}
} app;