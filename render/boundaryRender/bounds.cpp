#include <glad/glad.h>
#include <render/boundaryRender/bounds.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec4 posAttr;
in vec2 uvAttr;

uniform vec4 camera_right;
uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;
uniform vec3 vrtxDomainMin;
uniform vec3 vrtxDomainMax;
uniform vec3 render_clamp;
uniform float vrtxDomainEpsilon;

uniform sampler1D           colorRamp;

out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
flat out int invalid;

void main() {
	uv = uvAttr;
	//color = texture(colorRamp,renderIntensity);
	//color = vec4(renderIntensity,renderIntensity,renderIntensity,1.f);
	color = vec4(0.4,0.4,0.4,1.0);
	vec3 pos;
	pos.x = posAttr.x < 0.f ? vrtxDomainMin.x + 0.5f : vrtxDomainMax.x - 0.5f;
	pos.y = posAttr.y < 0.f ? vrtxDomainMin.y + 0.5f : vrtxDomainMax.y - 0.5f;
	pos.z = posAttr.z < 0.f ? vrtxDomainMin.z + 0.5f : vrtxDomainMax.z - 0.5f;
	
	pos.x = abs(pos.x) > 1e10f ? sign(pos.x) * 1e7f : pos.x;
	pos.y = abs(pos.y) > 1e10f ? sign(pos.y) * 1e7f : pos.y;
	pos.z = abs(pos.z) > 1e10f ? sign(pos.z) * 1e7f : pos.z;

	eyeSpacePos = view_matrix * vec4(pos.xyz ,1.f);
	//eyeSpacePos += vec4(posAttr.xyz * position.w*2.f,0.f);
	gl_Position = perspective_matrix * eyeSpacePos;

})";

static const char *fragmentShaderSource = R"(#version 450 
in vec2 uv;
in vec4 color;
in vec4 eyeSpacePos;
flat in int invalid;
out vec4 fragColor;
uniform sampler1D           colorRamp;
uniform mat4 perspective_matrix;

void main() {
	if(invalid == 1)
		discard;
	vec3 N;
	N.xy = uv.xy * vec2(2.f, 2.f) - vec2(1.f, 1.f);
	float r2 = dot(N.xy,N.xy);
	//if( r2 > 1.f) discard;
	N.z = sqrt(1.f - r2);

	vec3 lightDir = vec3(0, 0, 1);
	float diffuse = abs(dot(N, lightDir));

	fragColor = color;
	//gl_FragColor = color;
	//gl_FragColor = texture(colorRamp,uv.x);
	//gl_FragColor = vec4(N,1.f) ;

vec4 sphereEyeSpacePos;
sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
sphereEyeSpacePos.w = 1.0;
vec4 projPos = (perspective_matrix * sphereEyeSpacePos);
//gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool BoundsRenderer::valid() { return true; }

void BoundsRenderer::update() { colorMap::instance().update(); }

BoundsRenderer::BoundsRenderer() {
    m_program = createProgram(vertexShaderSource, fragmentShaderSource);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  m_posAttr = glGetAttribLocation(m_program, "posAttr");
  m_colAttr = glGetAttribLocation(m_program, "colorAttr");


  std::vector<uint32_t> idx = {// front
                               0, 1, 2, 3,
                               // top
                               1, 5, 6, 2,
                               // back
                               7, 6, 5, 4,
                               // bottom
                               4, 0, 3, 7,
                               // left
                               4, 5, 1, 0,
                               // right
                               3, 2, 6, 7};
  glGenBuffers(1, &IBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);
  glUseProgram(m_program);
  //parent->bind(m_program);

#define vtx000 -1.0, -1.0, -1.0
#define vtx001 -1.0, -1.0,  1.0
#define vtx010 -1.0,  1.0, -1.0
#define vtx011 -1.0,  1.0,  1.0
#define vtx100  1.0, -1.0, -1.0
#define vtx101  1.0, -1.0,  1.0
#define vtx110  1.0,  1.0, -1.0
#define vtx111  1.0,  1.0,  1.0

  GLfloat vertices[] = {
      vtx000, vtx001,
      vtx000, vtx010,
      vtx000, vtx100,
      vtx111, vtx011,
      vtx111, vtx101,
      vtx111, vtx110,
      vtx100, vtx110,
      vtx100, vtx101,
      vtx010, vtx011,
      vtx001, vtx011,
      vtx010, vtx110,
      vtx001, vtx101,
  };

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  GLfloat uvs[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 2, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  colorMap::instance().bind(m_program, 0, "colorRamp");
  update();
}
#include <utility/identifier/uniform.h>
void BoundsRenderer::render(bool pretty) {
	if (get<parameters::render_settings::boundsRender>() != 1) return;
  glBindVertexArray(vao);
  glUseProgram(m_program);
  //m_program->bind();
  glPolygonMode(GL_FRONT, GL_LINE);
  glPolygonMode(GL_BACK, GL_LINE);

  //int size;
  //glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
  glDrawArrays(GL_LINES, 0, 24);

  glPolygonMode(GL_FRONT, GL_FILL);
  glPolygonMode(GL_BACK, GL_FILL);
  //m_program->release();
  glUseProgram(0);
  glBindVertexArray(0);
}
