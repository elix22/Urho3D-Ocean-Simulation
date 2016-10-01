//=============================================================================
//=============================================================================

#pragma once

#include <Urho3D/Container/Vector.h>
#include <Urho3D/Core/Thread.h>

#include "ComplexFFT.h"

namespace Urho3D
{
class Material;
class Model;
class Timer;
}

using namespace Urho3D;

//=============================================================================
//=============================================================================
template<class T>
class HelperThread : public Thread
{
public:
    typedef void (T::*ProcessFn)();

    HelperThread(T *parent, ProcessFn pFn, bool looping=true, int priority=M_MAX_INT) 
        : parent_(parent), processFn_(pFn), looping_(looping), priority_(priority), fnExited_(true)
    {
    }

    virtual ~HelperThread()
    {
        WaitExit();
    }

    void Start()
    {
        Run();
        SetPriority(priority_);
    }

    virtual void ThreadFunction()
    {
        SetFnExit(false);

        while (true)
        {
            // process callback
            (parent_->*processFn_)();

            if (!IsLooping())
                break;

            Time::Sleep(0);
        }

        SetFnExit(true);
    }

    bool HasFnExited()
    {
        MutexLock lock(mutexLock_);
        return fnExited_;
    }

protected:
    void WaitExit()
    {
        SetLooping(false);

        do 
        { 
            Time::Sleep(1); 
        }
        while (!HasFnExited());
    }

    void SetLooping(bool bset)
    {
        MutexLock lock(mutexLock_);
        looping_ = bset;
    }

    bool IsLooping()
    {
        MutexLock lock(mutexLock_);
        return looping_;
    }

    void SetFnExit(bool bset)
    {
        MutexLock lock(mutexLock_);
        fnExited_ = bset;
    }

protected:
    T           *parent_;
    ProcessFn   processFn_;
    Mutex       mutexLock_;
    int         priority_;

    bool        looping_;
    bool        fnExited_;
};

//=============================================================================
//=============================================================================
struct vertex_ocean 
{
	float   x,   y,   z; // vertex
	float  nx,  ny,  nz; // normal
	float   a,   b,   c; // htilde0
	float  _a,  _b,  _c; // htilde0mk conjugate
	float  ox,  oy,  oz; // original position
};

// structure used with discrete fourier transform
struct complex_vector_normal 
{
	complex h; // wave height
	Vector2 D; // displacement
	Vector3 n; // normal
};


//=============================================================================
//=============================================================================
class cOcean {
  private:
	bool geometry;			// flag to render geometry or surface

	float g;				// gravity constant
	int N, Nplus1;			// dimension -- N should be a power of 2
	float A;				// phillips spectrum parameter -- affects heights of waves
	Vector2      w;			// wind parameter
	float length;			// length parameter
	complex *h_tilde,		// for fast fourier transform
		*h_tilde_slopex, *h_tilde_slopez,
		*h_tilde_dx, *h_tilde_dz;
	cFFT *fft;				// fast fourier transform

public:
	vertex_ocean *vertices;			// vertices for vertex buffer object
	unsigned int *indices;			// indicies for vertex buffer object
	unsigned int indices_count;		// number of indices to render
	//GLuint vbo_vertices, vbo_indices;	// vertex buffer objects

	//GLuint glProgram, glShaderV, glShaderF;	// shaders
	//GLint vertex, normal, texture, light_position, projection, view, model;	// attributes and uniforms

public:
	cOcean(const int N, const float A, const Vector2      w, const float length, bool geometry);
	~cOcean();
	void release();

	float dispersion(int n_prime, int m_prime);		// deep water
	float phillips(int n_prime, int m_prime);		// phillips spectrum
	complex hTilde_0(int n_prime, int m_prime);
	complex hTilde(float t, int n_prime, int m_prime);
	complex_vector_normal h_D_and_n(Vector2      x, float t);
	void evaluateWaves(float t);
	void evaluateWavesFFT(float t);
	//void render(float t, glm::vec3 light_pos, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, bool use_fft);
};

//=============================================================================
//=============================================================================
class Ocean : public Component
{
    URHO3D_OBJECT(Ocean, Component);

    struct Mesh
    {
        PODVector<Vector3> vertices;
        PODVector<Vector2> texcoords;
        PODVector<Vector3> normals;
        PODVector<int>     indices;
    };

public:
    static void RegisterObject(Context *context);

    Ocean(Context *context);
    ~Ocean();

    void InitOcean();

    Model* GetOceanModel() const        { return m_pModelOcean; }
    BoundingBox GetBoundingBox() const  { return m_BoundingBox; }

    void DbgRender();

protected:
    // fft
    void EvaluateWavesFFT();
    void UpdateVertexBuffer();
    void MakeMesh(int size, Mesh &mesh);

    // threading
    void SetProcessPending(bool bset);
    bool IsProcessPending();
    void BackgroundProcess();

    void HandleUpdate(StringHash eventType, VariantMap& eventData);

protected:
    // ocean
    cOcean *pCOcean;
    int     N;
    int     Nplus1;	

    Mesh             m_mesh;
    SharedPtr<Model> m_pModelOcean;
    BoundingBox      m_BoundingBox;

    // background thread
    HelperThread<Ocean> *threadProcess_;
    Mutex               mutexPendingLock_;
    bool                processPending_;
    SharedPtr<Time>     elapsedFrameTimer_;
    Timer               processTimer_;
};



