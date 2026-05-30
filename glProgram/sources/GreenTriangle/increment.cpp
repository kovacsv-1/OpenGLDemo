#include "framework.h"

float radians(float angle) {
    return angle * M_PI / 180.0f;
}

// === Camera & Material from your previous struct ===
struct Camera {
    vec3 wEye = vec3(0.0f, 1.0f, 4.0f), wLookat = vec3(0.0f, 0.0f, 0.0f), wVup = vec3(0.0f, 1.0f, 0.0f);
    float fov = radians(45.0f), asp = 1.0f, fp = 0.1f, bp = 100.0f;
    mat4 V() { return lookAt(wEye, wLookat, wVup); }
    mat4 P() { return perspective(fov, asp, fp, bp); }
    void set(vec3 eye, vec3 lookat, vec3 vup, float fov) {
        wEye = eye;
        wLookat = lookat;
        wVup = vup;
        this->fov = fov;
    }
    void Animate(float state) {

        vec3 eye = vec3(4 * sin(state * M_PI / 4), 1, 4 * cos(state * M_PI / 4)), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * (float)M_PI / 180;
        set(eye, lookat, vup, fov);
    }
};

struct Material {
    vec3 ka, kd, ks;
    float shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess)
        : ka(_kd * 3.0f), kd(_kd), ks(_ks), shininess(_shininess) {
    }
};

// === Vertex Format ===
struct Vertex {
    vec3 pos, normal;
};

template<>
class Geometry<Vertex> {
    GLuint vao, vbo;
    std::vector<Vertex> vtx;

public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0); // position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));

        glEnableVertexAttribArray(1); // normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    }

    std::vector<Vertex>& Vtx() { return vtx; }

    void updateGPU() {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(Vertex), &vtx[0], GL_DYNAMIC_DRAW);
    }

    void Bind() {
        glBindVertexArray(vao);
    }

    void Draw(GPUProgram* prog, int type, vec3 color) {
        if (vtx.size() > 0) {
            //prog->setUniform(color, "color");
            Bind();
            glDrawArrays(type, 0, (GLsizei)vtx.size());
        }
    }

    virtual ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

// === Geometry for Cone ===
class Cone : public Geometry<Vertex> {
public:
    Cone() {
        const int n = 12;
        const float r = 1.0f, h = 2.0f;
        vec3 apex(0, h / 2, 0);

        for (int i = 0; i < n; ++i) {
            float t0 = i * 2.0f * M_PI / n;
            float t1 = (i + 1) * 2.0f * M_PI / n;

            vec3 p0(r * cos(t0), -h / 2, r * sin(t0));
            vec3 p1(r * cos(t1), -h / 2, r * sin(t1));

            vec3 n = normalize(cross(p1 - apex, p0 - apex));

            Vtx().push_back(Vertex(apex, n));
            Vtx().push_back(Vertex(p0, n));
            Vtx().push_back(Vertex(p1, n));
        }

        updateGPU();
    }
};

// === Shader Sources ===
const char* vertexShader = R"(
#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;

uniform mat4 MVP, M;
out vec3 worldPos, worldNormal;

void main() {
    vec4 world = M * vec4(vertexPosition, 1.0);
    worldPos = vec3(world);
    worldNormal = mat3(transpose(inverse(M))) * vertexNormal;
    gl_Position = MVP * vec4(vertexPosition, 1.0);
}
)";

const char* fragmentShader = R"(
#version 330 core

in vec3 worldPos, worldNormal;
out vec4 fragColor;

uniform vec3 kd, ks, eyePos;
uniform float shininess;
uniform bool useChecker;

// Lighting
uniform vec3 lightDir;            // normalized, from light to surface
uniform float lightIntensity;     // e.g. 2.0
const vec3 ambientLight = vec3(1.0) * 0.4; // white ambient light

// Shadow setup
#define MAX_TRIS 256
uniform int triCount;
uniform vec3 triVerts[MAX_TRIS * 3];

// === Utility: Checker pattern ===
vec3 getCheckerColor(vec3 pos) {
    int x = int(floor(pos.x + 10.0));
    int z = int(floor(pos.z + 10.0));
    bool isWhite = ((x + z) % 2 == 0);
    return isWhite ? vec3(0.3) : vec3(0.0, 0.1, 0.3);
}

// === Utility: Möller–Trumbore triangle-ray intersection ===
bool intersectTriangle(vec3 orig, vec3 dir, vec3 v0, vec3 v1, vec3 v2, out float t) {
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
    t = f * dot(edge2, q);
    return t > EPSILON;
}

// === Utility: Shadow testing ===
bool isInShadow(vec3 pos, vec3 lightDir) {
    vec3 dir = normalize(lightDir);
    float t;
    for (int i = 0; i < triCount; ++i) {
        vec3 v0 = triVerts[i * 3 + 0];
        vec3 v1 = triVerts[i * 3 + 1];
        vec3 v2 = triVerts[i * 3 + 2];
        if (intersectTriangle(pos + dir * 0.01, dir, v0, v1, v2, t)) {
            return true;
        }
    }
    return false;
}

// === Main ===
void main() {
    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightDir); // light *toward* surface
    vec3 V = normalize(eyePos - worldPos);
    vec3 H = normalize(L + V);

    vec3 finalKd = useChecker ? getCheckerColor(worldPos) : kd;
    vec3 finalKa = finalKd * 3.0;
    vec3 finalKs = useChecker ? vec3(0.0) : ks;
    float finalShininess = useChecker ? 0.0 : shininess;

    vec3 ambient = finalKa * ambientLight;

    bool shadow = isInShadow(worldPos, lightDir);

    if (shadow) {
        fragColor = vec4(ambient, 1.0);
    } else {
        vec3 diffuse = finalKd * max(dot(N, L), 0.0) * lightIntensity;
        vec3 specular = finalKs * pow(max(dot(N, H), 0.0), finalShininess) * lightIntensity;
        vec3 finalColor = ambient + diffuse + specular;
        finalColor = pow(finalColor, vec3(1.0 / 2.2)); // gamma correction
        fragColor = vec4(finalColor, 1.0);
    }
}


)";

class OrientedCone : public Geometry<Vertex> {
public:
    OrientedCone(vec3 apex, vec3 axisDir, float angleRad, float height, int slices = 12) {
        vec3 axis = normalize(axisDir);
        vec3 baseCenter = apex + axis * height;
        float radius = tan(angleRad) * height;

        // Local coordinate system
        vec3 up = abs(dot(axis, vec3(0, 1, 0))) < 0.999f ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 right = normalize(cross(axis, up));
        vec3 forward = normalize(cross(right, axis));

        for (int i = 0; i < slices; ++i) {
            float t0 = (float)i / slices * 2.0f * M_PI;
            float t1 = (float)(i + 1) / slices * 2.0f * M_PI;

            // Circle points in local base plane
            vec3 dir0 = cos(t0) * right + sin(t0) * forward;
            vec3 dir1 = cos(t1) * right + sin(t1) * forward;

            vec3 p0 = baseCenter + dir0 * radius;
            vec3 p1 = baseCenter + dir1 * radius;

            // One triangle per side
            vec3 n = normalize(cross(p1 - apex, p0 - apex));

            Vtx().push_back(Vertex(apex, n));
            Vtx().push_back(Vertex(p0, n));
            Vtx().push_back(Vertex(p1, n));
        }

        updateGPU();
    }
};

class Cylinder : public Geometry<Vertex> {
public:
    Cylinder(vec3 base, vec3 axisDir, float radius, float height, int sides = 6) {
        vec3 axis = normalize(axisDir);
        vec3 top = base + axis * height;

        // Lokális koordináta-rendszer a körhöz
        vec3 up = abs(dot(axis, vec3(0, 1, 0))) < 0.999f ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 right = normalize(cross(axis, up));
        vec3 forward = normalize(cross(right, axis));

        for (int i = 0; i < sides; ++i) {
            float t0 = (float)i / sides * 2.0f * M_PI;
            float t1 = (float)(i + 1) / sides * 2.0f * M_PI;

            vec3 d0 = cos(t0) * right + sin(t0) * forward;
            vec3 d1 = cos(t1) * right + sin(t1) * forward;

            vec3 b0 = base + d0 * radius;
            vec3 b1 = base + d1 * radius;
            vec3 t0p = top + d0 * radius;
            vec3 t1p = top + d1 * radius;

            // Két háromszög az oldallaphoz
            vec3 n0 = normalize(cross(t1p - b0, b1 - b0));

            Vtx().push_back(Vertex(b0, n0));
            Vtx().push_back(Vertex(b1, n0));
            Vtx().push_back(Vertex(t1p, n0));

            vec3 n1 = normalize(cross(t0p - b0, t1p - b0));

            Vtx().push_back(Vertex(b0, n1));
            Vtx().push_back(Vertex(t1p, n1));
            Vtx().push_back(Vertex(t0p, n1));
        }

        updateGPU();
    }
};

struct PlaneVertex {
    vec3 pos;
    vec2 uv;
};

class Plane : public Geometry<Vertex> {
public:
    Plane() {
        float halfSize = 10.0f;
        float y = -1.0f;
        vec3 apex, p0, p1, n;

        apex = vec3(-halfSize, y, -halfSize);
        p0 = vec3(halfSize, y, -halfSize);
        p1 = vec3(halfSize, y, halfSize);
        n = normalize(cross(p1 - apex, p0 - apex));
        Vtx().push_back(Vertex(apex, n));
        Vtx().push_back(Vertex(p0, n));
        Vtx().push_back(Vertex(p1, n));

        apex = vec3(-halfSize, y, halfSize);
        p1 = vec3(halfSize, y, halfSize);
        p0 = vec3(-halfSize, y, -halfSize);
        n = normalize(cross(p1 - apex, p0 - apex));
        Vtx().push_back(Vertex(apex, n));
        Vtx().push_back(Vertex(p0, n));
        Vtx().push_back(Vertex(p1, n));

        updateGPU();
    }
};

// === Application ===
class MyApp : public glApp {
    GPUProgram shader;
    OrientedCone* cone1 = nullptr, * cone2 = nullptr;
    Camera cam;
    Cylinder* cylinder = nullptr;
    Cylinder* goldCylinder = nullptr, * waterCylinder = nullptr;
    Plane* groundPlane = nullptr;
    Material matGold = Material(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f), 100);
    Material matWater = Material(vec3(1.3f), vec3(0.0f), 100);
    Material mat3 = Material(vec3(0.3f, 0.2f, 0.1f), vec3(2.0f), 50);
    Material mat1 = Material(vec3(0.1f, 0.2f, 0.3f), vec3(2.0f), 100); // Cián
    Material mat2 = Material(vec3(0.3f, 0.0f, 0.2f), vec3(2.0f), 20);  // Magenta
    float angle = 0;
    int campos = 0;

public:
    MyApp() : glApp("Oriented Cones") {}

    void uploadTriangleData() {
        std::vector<vec3> tris;

        auto collectTriangles = [&](Geometry<Vertex>* geom, const mat4& M) {
            for (size_t i = 0; i + 2 < geom->Vtx().size(); i += 3) {
                // Correct vec4 construction
                vec4 vertex0 = vec4(geom->Vtx()[i + 0].pos.x, geom->Vtx()[i + 0].pos.y, geom->Vtx()[i + 0].pos.z, 1.0f);
                vec4 vertex1 = vec4(geom->Vtx()[i + 1].pos.x, geom->Vtx()[i + 1].pos.y, geom->Vtx()[i + 1].pos.z, 1.0f);
                vec4 vertex2 = vec4(geom->Vtx()[i + 2].pos.x, geom->Vtx()[i + 2].pos.y, geom->Vtx()[i + 2].pos.z, 1.0f);

                vec3 v0 = vec3((M * vertex0).x, (M * vertex0).y, (M * vertex0).z);
                vec3 v1 = vec3((M * vertex1).x, (M * vertex1).y, (M * vertex1).z);
                vec3 v2 = vec3((M * vertex2).x, (M * vertex2).y, (M * vertex2).z);

                tris.push_back(v0);
                tris.push_back(v1);
                tris.push_back(v2);
            }
            };

        mat4 identity = mat4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );
        if (cone1) collectTriangles(cone1, identity);
        if (cone2) collectTriangles(cone2, identity);
        if (cylinder) collectTriangles(cylinder, identity);
        if (goldCylinder) collectTriangles(goldCylinder, identity);
        if (waterCylinder) collectTriangles(waterCylinder, identity);
        if (groundPlane) collectTriangles(groundPlane, identity);

        GLint progId = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &progId);

        GLint locTriCount = glGetUniformLocation(progId, "triCount");
        GLint locTriVerts = glGetUniformLocation(progId, "triVerts");

        glUniform1i(locTriCount, (GLint)(tris.size() / 3));
        glUniform3fv(locTriVerts, tris.size(), &tris[0].x);
    }

    void onInitialization() override {
        glEnable(GL_DEPTH_TEST);
        shader.create(vertexShader, fragmentShader);

        // Kúp 1 (Cián)
        cone1 = new OrientedCone(
            vec3(0, 1, 0),                // csúcs
            normalize(vec3(-0.1f, -1, -0.05f)),
            0.2f,                         // nyílásszög (radian)
            2.0f                          // magasság
        );

        // Kúp 2 (Magenta)
        cone2 = new OrientedCone(
            vec3(0, 1, 0.8f),
            normalize(vec3(0.2f, -1, 0)),
            0.2f,
            2.0f
        );

        cylinder = new Cylinder(
            vec3(-1.0f, -1.0f, 0.0f),        // alappont
            normalize(vec3(0, 1, 0.1f)),     // tengelyirány
            0.3f,                            // sugár
            2.0f                             // magasság
        );

        goldCylinder = new Cylinder(
            vec3(1.0f, -1.0f, 0.0f),
            normalize(vec3(0.1f, 1.0f, 0.0f)),
            0.3f,
            2.0f
        );

        waterCylinder = new Cylinder(
            vec3(0.0f, -1.0f, -0.8f),
            normalize(vec3(-0.2f, 1.0f, -0.1f)),
            0.3f,
            2.0f
        );
        groundPlane = new Plane();
    }

    void onDisplay() override {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 V = cam.V();
        mat4 P = cam.P();

        shader.Use();
        shader.setUniform(cam.wEye, "eyePos");
        //shader.setUniform(vec3(5, 5, 5), "lightPos");
        shader.setUniform(normalize(vec3(1, 1, 1)), "lightDir");
        shader.setUniform(2.0f, "lightIntensity");

        uploadTriangleData();

        // --- Cián kúp ---
        mat4 MI = mat4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );
        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(mat1.ka, "ka");
        shader.setUniform(mat1.kd, "kd");
        shader.setUniform(mat1.ks, "ks");
        shader.setUniform(mat1.shininess, "shininess");
        cone1->Draw(&shader, GL_TRIANGLES, mat1.kd);

        // --- Magenta kúp ---
        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(mat2.ka, "ka");
        shader.setUniform(mat2.kd, "kd");
        shader.setUniform(mat2.ks, "ks");
        shader.setUniform(mat2.shininess, "shininess");
        cone2->Draw(&shader, GL_TRIANGLES, mat2.kd);

        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(mat3.ka, "ka");
        shader.setUniform(mat3.kd, "kd");
        shader.setUniform(mat3.ks, "ks");
        shader.setUniform(mat3.shininess, "shininess");

        cylinder->Draw(&shader, GL_TRIANGLES, mat3.kd);

        // --- Arany henger ---
        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(matGold.ka, "ka");
        shader.setUniform(matGold.kd, "kd");
        shader.setUniform(matGold.ks, "ks");
        shader.setUniform(matGold.shininess, "shininess");
        goldCylinder->Draw(&shader, GL_TRIANGLES, matGold.kd);

        // --- Víz henger ---
        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(matWater.ka, "ka");
        shader.setUniform(matWater.kd, "kd");
        shader.setUniform(matWater.ks, "ks");
        shader.setUniform(matWater.shininess, "shininess");
        waterCylinder->Draw(&shader, GL_TRIANGLES, matWater.kd);

        // --- Sakktábla sík ---
        shader.setUniform(P * V * MI, "MVP");
        shader.setUniform(MI, "M");
        //shader.setUniform(V, "V");
        shader.setUniform(true, "useChecker");// Nem használjuk material structot, hanem shaderben számoljuk ki
        groundPlane->Draw(&shader, GL_TRIANGLES, vec3(1, 1, 1));
        shader.setUniform(false, "useChecker");

    }

    void onKeyboard(int key) {
        if (key == 'a') {
            campos = (campos + 1) % 8;
            cam.Animate(campos);
            refreshScreen();
        }
    }

    ~MyApp() {
        delete cone1;
        delete cone2;
        delete goldCylinder;
        delete waterCylinder;
        delete cylinder;
        delete groundPlane;
    }
} App;