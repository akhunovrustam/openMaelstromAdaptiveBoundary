#include <glad/glad.h>
#include <vector>
#include <render/util/renderer.h>
#include <iostream>

GLuint createProgram(std::string vertexSource, std::string fragmentSource) {
  // Create an empty vertex shader handle
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

  // Send the vertex shader source code to GL
  // Note that std::string's .c_str is NULL character terminated.
  const GLchar *source = (const GLchar *)vertexSource.c_str();
  glShaderSource(vertexShader, 1, &source, 0);

  // Compile the vertex shader
  glCompileShader(vertexShader);

  GLint isCompiled = 0;
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &isCompiled);
  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> infoLog(maxLength);
    glGetShaderInfoLog(vertexShader, maxLength, &maxLength, &infoLog[0]);

    // We don't need the shader anymore.
    glDeleteShader(vertexShader);

    // Use the infoLog as you see fit.
    std::cerr << infoLog.data() << std::endl;

    // In this simple program, we'll just leave
    return -1;
  }

  // Create an empty fragment shader handle
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

  // Send the fragment shader source code to GL
  // Note that std::string's .c_str is NULL character terminated.
  source = (const GLchar *)fragmentSource.c_str();
  glShaderSource(fragmentShader, 1, &source, 0);

  // Compile the fragment shader
  glCompileShader(fragmentShader);

  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &isCompiled);
  if (isCompiled == GL_FALSE) {
    GLint maxLength = 0;
    glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> infoLog(maxLength);
    glGetShaderInfoLog(fragmentShader, maxLength, &maxLength, &infoLog[0]);

    // We don't need the shader anymore.
    glDeleteShader(fragmentShader);
    // Either of them. Don't leak shaders.
    glDeleteShader(vertexShader);
    std::cerr << infoLog.data() << std::endl;

    // Use the infoLog as you see fit.

    // In this simple program, we'll just leave
    return -1;
  }

  // Vertex and fragment shaders are successfully compiled.
  // Now time to link them together into a program.
  // Get a program object.
  GLuint program = glCreateProgram();

  // Attach our shaders to our program
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  // Link our program
  glLinkProgram(program);

  // Note the different functions here: glGetProgram* instead of glGetShader*.
  GLint isLinked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, (int *)&isLinked);
  if (isLinked == GL_FALSE) {
    GLint maxLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

    // The maxLength includes the NULL character
    std::vector<GLchar> infoLog(maxLength);
    glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
    std::cerr << infoLog.data() << std::endl;

    // We don't need the program anymore.
    glDeleteProgram(program);
    // Don't leak shaders either.
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Use the infoLog as you see fit.

    // In this simple program, we'll just leave
    return -1;
  }

  // Always detach shaders after a successful link.
  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);
  return program;
}

#define BOOST_USE_WINDOWS_H
#include <iostream>
#include <utility/helpers/arguments.h>
#include <utility/identifier/uniform.h>
#include <render/util/renderer.h>
#include <render/boundaryRender/bounds.h>
#include <render/particleRender/particle_render.h>
#include <render/volumeRender/volume_render.h>
#include <math/template/tuple_for_each.h>
#include <boost/format.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <chrono>
#include <mutex>
#include <simulation/particleSystem.h>
#include <sstream>
#include <utility/helpers/arguments.h>
#include <tools/timer.h>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <math/math.h>

RTXRender::RTXRender() {
	if (get<parameters::modules::rayTracing>() == false) return;
	static const char *vertexShaderSource = R"(#version 450
in vec3 vertexPosition_modelspace;
out vec2 UV;

void main(){
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
	UV.y = 1.f - UV.y;
}
)";

	static const char *fragmentShaderSource = R"(#version 450 
uniform sampler2D renderedTexture;

in vec2 UV;
out vec4 color;

void main(){
	vec4 col = texture( renderedTexture, UV);
    //vec4 col = vec4(UV.x, UV.y, 0,1);
	color = vec4(col.xyz,1) ;
	//gl_FragDepth = col.w;
}
)";

	initialFrame = get<parameters::internal::frame>();
	auto h_scene = hostScene();
	cudaMalloc(&accumulatebuffer, h_scene.width * h_scene.height * sizeof(float3));
    quad_programID = createProgram(vertexShaderSource, fragmentShaderSource);

	glGenVertexArrays(1, &defer_VAO);
	glBindVertexArray(defer_VAO);
    glUseProgram(quad_programID);

	auto m_posAttr = glGetAttribLocation(quad_programID, "vertexPosition_modelspace");
	glGenTextures(1, &renderedTextureOut);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, h_scene.width, h_scene.height, 0, GL_RGBA, GL_FLOAT, 0);
	cudaGraphicsGLRegisterImage(&renderedResourceOut, renderedTextureOut, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsSurfaceLoadStore);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE0);
	static const GLfloat g_quad_vertex_bufferdata[] = {
		-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
	};

	GLuint quad_vertexbuffer;
	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_bufferdata), g_quad_vertex_bufferdata, GL_STATIC_DRAW);

	glEnableVertexAttribArray(m_posAttr);
	glVertexAttribPointer(m_posAttr, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
    glUniform1i(glGetUniformLocation(quad_programID, "renderedTexture"), 0);
	//quad_programID->setUniformValue("renderedTexture", 0);

	//quad_programID->release();
    glUseProgram(0);
	//prepCUDAscene();
	//update();

}

void RTXRender::update() {
	updateRTX();
}
bool RTXRender::valid() { 
	return bValid; 
}
void RTXRender::render(bool pretty) {
	static std::random_device r;
	static std::default_random_engine e1(r());
	static std::uniform_int_distribution<int32_t> uniform_dist(INT_MIN, INT_MAX);

	if (get<parameters::internal::frame>() == initialFrame) {
		return;
	}
	if (get<parameters::modules::rayTracing>() == false) return;
	auto h_scene = hostScene();
	if (h_scene.dirty || frame != get<parameters::internal::frame>() || dirty) {
		frame = get<parameters::internal::frame>();
		cudaMemset(accumulatebuffer, 1, h_scene.width * h_scene.height * sizeof(float3));
		framenumber = 0;
		dirty = false;
	}
	if (!bValid) return;
	int32_t iterations = 1;
	if (pretty)
		iterations = 50;

	for (int32_t i = 0; i < iterations; ++i) {
		framenumber++;
		renderRTX(pretty, framenumber, uniform_dist(e1));
	}
		glBindVertexArray(defer_VAO);
		//quad_programID->bind();
        glUseProgram(quad_programID);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, renderedTextureOut);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
		//quad_programID->release();
        glUseProgram(0);
		glBindVertexArray(0);
}
void RTXRender::prepCUDAscene() {

}


SceneInformation& hostScene() {
    static SceneInformation m_instance;
    return m_instance;
}