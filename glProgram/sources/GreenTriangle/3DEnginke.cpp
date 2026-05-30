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
class GouraudShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
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

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = MVP * vec4(vtxPos, 1.0); // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource/*, "fragmentColor"*/); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
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

		void main() {
			gl_Position = MVP * vec4(vtxPos, 1.0); // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
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

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
			//vec3 ka = material.ka;
			//vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
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

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = MVP * vec4(vtxPos, 1.0); // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource/*, "fragmentColor"*/); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		int textureUnit = 0; // textúra mintavevő egység
		setUniform(textureUnit, "diffuseTexture");
		if (state.texture != nullptr) {
			state.texture->Bind(textureUnit);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}
		setUniform(state.lights[0].wLightPos, "wLightPos");
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

public:
	Object3D() {
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
	}

	virtual ~Object3D() {
		if (vbo) glDeleteBuffers(1, &vbo);
		if (vao) glDeleteVertexArrays(1, &vao);
	}

	void uploadVertexData(const std::vector<VertexData>& vertices) {
		vertexCount = vertices.size();
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
	Quad() {
		std::vector<VertexData> verts = {
			{ vec3(-1,-1,0), vec3(0,0,1), vec2(0,0) },
			{ vec3(1,-1,0), vec3(0,0,1), vec2(1,0) },
			{ vec3(1,1,0), vec3(0,0,1), vec2(1,1) },

			{ vec3(-1,-1,0), vec3(0,0,1), vec2(0,0) },
			{ vec3(1,1,0), vec3(0,0,1), vec2(1,1) },
			{ vec3(-1,1,0), vec3(0,0,1), vec2(0,1) }
		};
		uploadVertexData(verts);
	}
};

class LowPolyCylinder : public Object3D {
public:
	LowPolyCylinder() {
		const int segments = 6;
		const float r = 1.0f, h = 2.0f;
		std::vector<VertexData> verts;

		for (int i = 0; i < segments; ++i) {
			float a0 = (float)i / segments * 2.0f * M_PI;
			float a1 = (float)(i + 1) / segments * 2.0f * M_PI;

			vec3 p0(r * cos(a0), -h / 2, r * sin(a0));
			vec3 p1(r * cos(a1), -h / 2, r * sin(a1));
			vec3 p2(r * cos(a0), h / 2, r * sin(a0));
			vec3 p3(r * cos(a1), h / 2, r * sin(a1));

			vec3 n0 = normalize(vec3(cos(a0), 0, sin(a0)));
			vec3 n1 = normalize(vec3(cos(a1), 0, sin(a1)));

			// oldal 1
			verts.push_back({ p0, n0, vec2((float)i / segments, 0) });
			verts.push_back({ p1, n1, vec2((float)(i + 1) / segments, 0) });
			verts.push_back({ p2, n0, vec2((float)i / segments, 1) });

			// oldal 2
			verts.push_back({ p1, n1, vec2((float)(i + 1) / segments, 0) });
			verts.push_back({ p3, n1, vec2((float)(i + 1) / segments, 1) });
			verts.push_back({ p2, n0, vec2((float)i / segments, 1) });
		}

		uploadVertexData(verts); // 12 háromszög = 36 vertex
	}
};

class Cone : public Object3D {
public:
	Cone() {
		const int segments = 6;
		const float r = 1.0f, h = 2.0f;
		vec3 tip(0, h / 2, 0);
		vec3 baseCenter(0, -h / 2, 0);
		std::vector<VertexData> verts;

		for (int i = 0; i < segments; ++i) {
			float a0 = (float)i / segments * 2.0f * M_PI;
			float a1 = (float)(i + 1) / segments * 2.0f * M_PI;

			vec3 p0(r * cos(a0), -h / 2, r * sin(a0));
			vec3 p1(r * cos(a1), -h / 2, r * sin(a1));

			vec3 sideNormal = normalize(cross(p1 - tip, p0 - tip));

			// oldal
			verts.push_back({ tip, sideNormal, vec2(0.5f, 1.0f) });
			verts.push_back({ p0, sideNormal, vec2((float)i / segments, 0.0f) });
			verts.push_back({ p1, sideNormal, vec2((float)(i + 1) / segments, 0.0f) });

			// alap
			vec3 n = vec3(0, -1, 0);
			verts.push_back({ baseCenter, n, vec2(0.5f, 0.5f) });
			verts.push_back({ p1, n, vec2(0.5f + 0.5f * cos(a1), 0.5f + 0.5f * sin(a1)) });
			verts.push_back({ p0, n, vec2(0.5f + 0.5f * cos(a0), 0.5f + 0.5f * sin(a0)) });
		}

		uploadVertexData(verts); // 12 háromszög = 36 vertex
	}
};


//---------------------------
class ParamSurface : public Geometry<VertexData> {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = normalize(cross(drdU, drdV)); // normal vector
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		Bind();
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

//---------------------------
class Tractricoid : public ParamSurface {
	//---------------------------
public:
	Tractricoid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float height = 3.0f;
		U = U * height, V = V * 2 * M_PI;
		X = Cos(V) / Cosh(U); Y = Sin(V) / Cosh(U); Z = U - Tanh(U);
	}
};

//---------------------------
class Cylinder : public ParamSurface {
	//---------------------------
public:
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * 2 - 1.0f;
		X = Cos(U); Z = Sin(U); Y = V;
	}
};

//---------------------------
class Torus : public ParamSurface {
	//---------------------------
public:
	Torus() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float R = 1, r = 0.5f;
		U = U * 2.0f * M_PI, V = V * 2.0f * M_PI;
		Dnum2 D = Cos(U) * r + R;
		X = D * Cos(V); Y = D * Sin(V); Z = Sin(U) * r;
	}
};

//---------------------------
class Mobius : public ParamSurface {
	//---------------------------
public:
	Mobius() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float R = 1, width = 0.5f;
		U = U * M_PI, V = (V - 0.5f) * width;
		X = (Cos(U) * V + R) * Cos(U * 2);
		Y = (Cos(U) * V + R) * Sin(U * 2);
		Z = Sin(U) * V;
	}
};

//---------------------------
class Klein : public ParamSurface {
	//---------------------------
	const float size = 1.5f;
public:
	Klein() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * M_PI * 2, V = V * M_PI * 2;
		Dnum2 a = Cos(U) * (Sin(U) + 1) * 0.3f;
		Dnum2 b = Sin(U) * 0.8f;
		Dnum2 c = (Cos(U) * (-0.1f) + 0.2f);
		X = a + c * ((U.f > M_PI) ? Cos(V + M_PI) : Cos(U) * Cos(V));
		Y = b + ((U.f > M_PI) ? 0 : c * Sin(U) * Cos(V));
		Z = c * Sin(V);
	}
};

//---------------------------
class Boy : public ParamSurface {
	//---------------------------
public:
	Boy() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = (U - 0.5f) * M_PI, V = V * M_PI;
		float r2 = sqrt(2.0f);
		Dnum2 denom = (Sin(U * 3) * Sin(V * 2) * (-3 / r2) + 3) * 1.2f;
		Dnum2 CosV2 = Cos(V) * Cos(V);
		X = (Cos(U * 2) * CosV2 * r2 + Cos(U) * Sin(V * 2)) / denom;
		Y = (Sin(U * 2) * CosV2 * r2 - Sin(U) * Sin(V * 2)) / denom;
		Z = (CosV2 * 3) / denom;
	}
};

//---------------------------
class Dini : public ParamSurface {
	//---------------------------
	Dnum2 a = 1.0f, b = 0.15f;
public:
	Dini() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 4 * M_PI, V = V * (1 - 0.1f) + 0.1f;
		X = a * Cos(U) * Sin(V);
		Y = a * Sin(U) * Sin(V);
		Z = a * (Cos(V) + Log(Tan(V / 2))) + b * U + 3;
	}
};

//---------------------------
class Paraboloid : public ParamSurface {
	//---------------------------
	Dnum2 a = 1.0f, b = 0.15f;
public:
	Paraboloid() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 * M_PI;
		V = V * 2;
		X = V * Cos(U);
		Z = V * Sin(U);
		Y = X * X + Z * Z;
	}
};

//---------------------------
class Hyperboloid : public ParamSurface {
	//---------------------------
public:
	Hyperboloid() { create(); }

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		V = V * 2 * M_PI;
		U = U * 2 - 1;
		X = Cosh(U) * Cos(V);
		Z = Cosh(U) * Sin(V);
		Y = Sinh(U);
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	ParamSurface* geometry;
	Object3D* geometry3D;
	vec3 scaling, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, ParamSurface* _geometry) :
		scaling(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
		geometry3D = nullptr;
	}

	Object(Shader* _shader, Material* _material, Texture* _texture, Object3D* _geometry3D) :
		scaling(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = nullptr;
		geometry3D = _geometry3D;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = translate(translation) * rotate(rotationAngle, rotationAxis) * scale(scaling);
		Minv = scale(vec3(1 / scaling.x, 1 / scaling.y, 1 / scaling.z)) * rotate(-rotationAngle, rotationAxis) * translate(-translation);
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
		if (geometry) geometry->Draw();
		else if (geometry3D) geometry3D->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera; // 3D camera
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();
		Shader* gouraudShader = new GouraudShader();
		Shader* nprShader = new NPRShader();

		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(1, 1, 0);
		material0->ks = vec3(0, 0, 0);
		material0->ka = vec3(0.5f, 0.5f, 0);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		// Textures
		Texture* texture4x8 = new Texture(4, 8);
		Texture* texture15x20 = new Texture(15, 20);

		// Geometries
		ParamSurface* sphere = new Sphere();
		ParamSurface* cylinder = new Cylinder();
		ParamSurface* tractricoid = new Tractricoid();
		ParamSurface* torus = new Torus();
		ParamSurface* mobius = new Mobius();
		ParamSurface* klein = new Klein();
		ParamSurface* boy = new Boy();
		ParamSurface* dini = new Dini();
		ParamSurface* paraboloid = new Paraboloid();
		ParamSurface* hyperboloid = new Hyperboloid();

		// Create objects by setting up their vertex data on the GPU
		Object* sphereObject1 = new Object(phongShader, material0, nullptr, sphere);
		sphereObject1->translation = vec3(-9, 3, 0);
		sphereObject1->scaling = vec3(1, 1, 1);
		sphereObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(sphereObject1);

		Object* myCone = new Object(phongShader, material1, nullptr, new Cone());
		myCone->translation = vec3(6, 0, 0);
		objects.push_back(myCone);
		objects.push_back(new Object(phongShader, material1, nullptr, new LowPolyCylinder()));
		objects.push_back(new Object(phongShader, material1, nullptr, new Quad()));


		// Create objects by setting up their vertex data on the GPU
		Object* CylinderObject1 = new Object(phongShader, material0, nullptr, cylinder);
		CylinderObject1->translation = vec3(-12, 3, 0);
		CylinderObject1->scaling = vec3(1, 1, 1);
		CylinderObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(CylinderObject1);

		// Create objects by setting up their vertex data on the GPU
		Object* tractiObject1 = new Object(phongShader, material0, texture15x20, tractricoid);
		tractiObject1->translation = vec3(-6, 3, 0);
		tractiObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(tractiObject1);

		Object* torusObject1 = new Object(phongShader, material0, texture4x8, torus);
		torusObject1->translation = vec3(-3, 3, 0);
		torusObject1->scaling = vec3(0.7f, 0.7f, 0.7f);
		torusObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(torusObject1);

		Object* mobiusObject1 = new Object(phongShader, material0, texture4x8, mobius);
		mobiusObject1->translation = vec3(0, 3, 0);
		mobiusObject1->scaling = vec3(0.7f, 0.7f, 0.7f);
		mobiusObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(mobiusObject1);

		Object* kleinObject1 = new Object(phongShader, material1, texture4x8, klein);
		kleinObject1->translation = vec3(3, 3, 0);
		kleinObject1->scaling = vec3(0.7f, 0.7f, 0.7f);
		kleinObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(kleinObject1);

		Object* boyObject1 = new Object(phongShader, material1, texture15x20, boy);
		boyObject1->translation = vec3(6, 3, 0);
		boyObject1->scaling = vec3(0.7f, 0.7f, 0.7f);
		boyObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(boyObject1);

		Object* diniObject1 = new Object(phongShader, material1, texture15x20, dini);
		diniObject1->translation = vec3(9, 3, 0);
		diniObject1->scaling = vec3(0.7f, 0.7f, 0.7f);
		diniObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(diniObject1);

				int nObjects = objects.size();
				for (int i = 0; i < nObjects; i++) {
					Object * object = new Object(*objects[i]);
					object->translation.y -= 3;
					object->shader = gouraudShader;
					objects.push_back(object);
					object = new Object(*objects[i]);
					object->translation.y -= 6;
					object->shader = nprShader;
					objects.push_back(object);
				}

				// Camera
		camera.wEye = vec3(0, 2, 30);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(1, 1, 1, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.4, 0.4, 0.4);
		lights[0].Le = vec3(1, 1, 1);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

class EngineApp : public glApp {
	Scene scene;
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
	void onTimeElapsed(float tstart, float tend) {
		scene.Animate(tstart, tend);
		refreshScreen();
	}
} app;