#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <Windows.h>
#include <string>

#include "cuda_gl_interop.h"

#include "Shaders/Shader.h"


#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "ScreenRenderer/Screen.h"
#include "Scene/Scene.h"
#include "ScreenRenderer/Camera3D.h"

bool Settings = false;
bool wasReleased = true;
bool wasPressed = false;


void getDesktopResolution(int* width, int* height)
{
	RECT desktop;

	const HWND hDesktop = GetDesktopWindow();

	GetWindowRect(hDesktop, &desktop);

	*width = desktop.right;
	*height = desktop.bottom;
}

void SetupWindow(GLFWwindow*& window);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
// settings
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 900;

//// camera3D
Camera3D camera;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;


// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// lighting
glm::vec3 lightDir(1.2f, 1.0f, 2.0f);

glm::vec3 verts[] = {
	{-1.0f,-1.0f, -10.0f},
	{1.0f,-1.0f, -10.0f},
	{0.0f, 1.0f, -10.0f}
};

int inds[] = {
	0, 1, 2
};

__device__ glm::vec3* deviceVertices;


Screen screen;
Scene scene;
int numOfSpheres = 1000;
float boxSize = 100.0f;

int numOfLights = 10;


int main()
{
	GLFWwindow* window;
	SetupWindow(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glEnable(GL_DEPTH_TEST);


	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
	ImGui::StyleColorsDark();


	screen.Init(SCR_WIDTH, SCR_HEIGHT);

	float currentFrame;
	float FPS = 0.0f;

	ImGuiStyle* style = &ImGui::GetStyle();



	SetUpRandomLights(scene, numOfLights, boxSize);
	SetUpRandomSpheres(scene, numOfSpheres, 3.0f, boxSize);
	camera.Position = glm::vec3(0, 0, 2.0f * boxSize);


	lastFrame = static_cast<float>(glfwGetTime());
	while (!glfwWindowShouldClose(window))
	{
		currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		FPS = 1 / deltaTime;

		processInput(window);

		if (Settings)
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			glfwSetCursorPosCallback(window, NULL);
			firstMouse = true;
		}
		else
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			glfwSetCursorPosCallback(window, mouse_callback);
		}


		glClearColor(0.043f, 0.067f, 0.494f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		mat3 fur = camera.GetFURMatrix();
		float tanHalfFov = glm::tan(camera.Zoom / 2);
		screen.RenderSpheres(scene, camera.Position, fur, tanHalfFov);




		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Value("FPS", FPS);
		ImGui::Value("Frame Time", deltaTime);
		ImGui::Value("Width", screen.width);
		ImGui::Value("Height", screen.height);
		ImGui::SliderInt("Number Of Spheres", &numOfSpheres, 0, 5000);
		if (ImGui::Button("Randomize spheres (Press R)"))
		{
			SetUpRandomSpheres(scene, numOfSpheres, 3.0f, boxSize);
		}
		ImGui::SliderInt("Number Of Lights", &numOfLights, 0, 50);
		if (ImGui::Button("Randomize lights (Press T)"))
		{
			SetUpRandomLights(scene, numOfLights, boxSize);
		}
		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());



		glfwSwapBuffers(window);
		glfwSwapInterval(0);
		glfwPollEvents();
	}
	glfwTerminate();
	Free(scene);
	return 0;
}



void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		camera.ProcessKeyboard(Camera_Movement3D::DOWN, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && wasReleased)
	{
		if (Settings == true) Settings = false;
		else Settings = true;
		wasReleased = false;
	}
	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_RELEASE)
		wasReleased = true;
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
	{
		SetUpRandomSpheres(scene, numOfSpheres, 3.0f, boxSize);
	}
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
	{
		SetUpRandomLights(scene, numOfLights, boxSize);
	}
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	screen.Resize(width, height);
	glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void SetupWindow(GLFWwindow*& window)
{
	int err = glfwInit();
	if (GLFW_FALSE)
	{
		std::cout << "Error" << std::endl;
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	int count;
	GLFWmonitor** monitors = glfwGetMonitors(&count);
	GLFWmonitor* monitor = monitors[count - 1];
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Boids", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);
	int width;
	int height;
	getDesktopResolution(&width, &height);
	glfwSetWindowPos(window, width / 2 - SCR_WIDTH / 2, height / 2 - SCR_HEIGHT / 2);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}