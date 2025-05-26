#include <glad/glad.h>
#include <render/axisRender/axes.h>
#include <render/util/color_map.h>

static const char *vertexShaderSource = R"(#version 450 
in vec3 posAttr;
in vec3 colorAttr;

//uniform vec4 camera_right;
//uniform vec4 camera_up;
uniform mat4 perspective_matrix;
uniform mat4 view_matrix;
//uniform vec3 minCoord;
//uniform vec3 maxCoord;
//uniform vec3 render_clamp;

//uniform float axesScale;

//uniform sampler1D           colorRamp;

//out vec2 uv;
out vec4 color;
out vec4 eyeSpacePos;
//flat out int invalid;


void main() {
	color = vec4(colorAttr.xyz,1.0);
	//color =vec4(1,0,0,1);
	eyeSpacePos = view_matrix * vec4(posAttr.xyz * 0.5f ,1.f);
	gl_Position = perspective_matrix * eyeSpacePos;

})";

static const char *fragmentShaderSource = R"(#version 450 
//in vec2 uv;
in vec4 color;
in vec4 eyeSpacePos;
//flat in int invalid;
out vec4 fragColor;
//uniform sampler1D           colorRamp;
uniform mat4 perspective_matrix;

void main() {
	fragColor = color;

	vec4 projPos = (perspective_matrix * eyeSpacePos);
	gl_FragDepth = (projPos.z / projPos.w)* 0.5 + 0.5;
})";

bool AxesRenderer::valid() { return true; }

void AxesRenderer::update() { colorMap::instance().update(); }

AxesRenderer::AxesRenderer() {
  m_program = createProgram(vertexShaderSource, fragmentShaderSource);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glUseProgram(m_program);
  m_posAttr = glGetAttribLocation(m_program, "posAttr");
  m_colAttr = glGetAttribLocation(m_program, "colorAttr");

  GLfloat vertices[] = {
	  // X-Axis
	-4.0, 0.0f, 0.0f,
	4.0, 0.0f, 0.0f,
	// arrow
	4.0, 0.0f, 0.0f,
	3.0, 1.0f, 0.0f,
	4.0, 0.0f, 0.0f,
	3.0, -1.0f, 0.0f,
	// Y-Axis
	0.0, -4.0f, 0.0f,
	0.0, 4.0f, 0.0f,
	// arrow
	0.0, 4.0f, 0.0f,
	1.0, 3.0f, 0.0f,
	0.0, 4.0f, 0.0f,
	-1.0, 3.0f, 0.0f,
	// Z-Axis
	0.0, 0.0f ,-4.0f,
	0.0, 0.0f ,4.0f,
	// arrow
	0.0, 0.0f ,4.0f,
	0.0, 1.0f ,3.0f,
	0.0, 0.0f ,4.0f,
	0.0, -1.0f ,3.0f
  };

  glGenBuffers(1, &VXO);
  glBindBuffer(GL_ARRAY_BUFFER, VXO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_posAttr);
  glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#define XCOLOR 1.f, 0.f, 0.f
#define YCOLOR 0.f, 1.f, 0.f
#define ZCOLOR 0.f, 0.f, 1.f
#define REPEAT6(x) x, x, x, x, x, x
  GLfloat uvs[] = {
	  REPEAT6(XCOLOR),
	  REPEAT6(YCOLOR),
	  REPEAT6(ZCOLOR)};

  glGenBuffers(1, &VUV);
  glBindBuffer(GL_ARRAY_BUFFER, VUV);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

  glEnableVertexAttribArray(m_colAttr);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
  update();
}
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
void AxesRenderer::render(bool pretty) {
	//if (get<parameters::render_settings::axesRender>() != 1) return;
  glBindVertexArray(vao);
  glUseProgram(m_program);
  //m_program->bind();
  glPolygonMode(GL_FRONT, GL_LINE);
  glPolygonMode(GL_BACK, GL_LINE);

  //Camera::instance().matrices.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  //Camera::instance().matrices.perspective = glm::perspective(glm::radians(45.f), 1920.f / 1080.f, 0.1f, 100.f);

  //std::cout << "View Matrix2: " << std::endl
  //    << Camera::instance().matrices.view[0][0] << " " << Camera::instance().matrices.view[0][1] << " " << Camera::instance().matrices.view[0][2] << " " << Camera::instance().matrices.view[0][3] << "\n"
  //    << Camera::instance().matrices.view[1][0] << " " << Camera::instance().matrices.view[1][1] << " " << Camera::instance().matrices.view[1][2] << " " << Camera::instance().matrices.view[1][3] << "\n"
  //    << Camera::instance().matrices.view[2][0] << " " << Camera::instance().matrices.view[2][1] << " " << Camera::instance().matrices.view[2][2] << " " << Camera::instance().matrices.view[2][3] << "\n"
  //    << Camera::instance().matrices.view[3][0] << " " << Camera::instance().matrices.view[3][1] << " " << Camera::instance().matrices.view[3][2] << " " << Camera::instance().matrices.view[3][3] << "\n";

  //std::cout << "Projection Matrix2: " << std::endl
  //    << Camera::instance().matrices.perspective[0][0] << " " << Camera::instance().matrices.perspective[0][1] << " " << Camera::instance().matrices.perspective[0][2] << " " << Camera::instance().matrices.perspective[0][3] << "\n"
  //    << Camera::instance().matrices.perspective[1][0] << " " << Camera::instance().matrices.perspective[1][1] << " " << Camera::instance().matrices.perspective[1][2] << " " << Camera::instance().matrices.perspective[1][3] << "\n"
  //    << Camera::instance().matrices.perspective[2][0] << " " << Camera::instance().matrices.perspective[2][1] << " " << Camera::instance().matrices.perspective[2][2] << " " << Camera::instance().matrices.perspective[2][3] << "\n"
  //    << Camera::instance().matrices.perspective[3][0] << " " << Camera::instance().matrices.perspective[3][1] << " " << Camera::instance().matrices.perspective[3][2] << " " << Camera::instance().matrices.perspective[3][3] << "\n";

  //glUniformMatrix4fv(glGetUniformLocation(m_program, "perspective_matrix"), 1, GL_FALSE, &Camera::instance().matrices.perspective[0][0]);
  //glUniformMatrix4fv(glGetUniformLocation(m_program, "view_matrix"), 1, GL_FALSE, &Camera::instance().matrices.view[0][0]);

  glDrawArrays(GL_LINES,0,18);

  glPolygonMode(GL_FRONT, GL_FILL);
  glPolygonMode(GL_BACK, GL_FILL);
  //m_program->release();
  glUseProgram(0);
  glBindVertexArray(0);
}
