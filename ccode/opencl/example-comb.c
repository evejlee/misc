/* this one is timing.  This is more optimized than 2*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/opencl.h>
#include "fmath.h"

//simulate 3 gaussians by doing the calculations 3 times
const char *kernel_source =
"__kernel void gmix(int nelem, \n"
"                   int ncol,  \n"
"                   float cenrow, \n"
"                   float cencol, \n"
"                   float idet, \n"
"                   float irr, \n"
"                   float irc, \n"
"                   float icc, \n"
"                   __constant float *image, \n"
"                   __constant float *rows, \n"
"                   __constant float *cols, \n"
"                   global float *output)                            \n"
"{                                                                     \n"
"   int idx = get_global_id(0);                                        \n"
"   if (idx >= nelem)                            \n"
"       return;                                  \n"
"   float tmp=0;                                 \n"
"   float row = rows[idx];                       \n"
"   float col = cols[idx];                       \n"
"   int im_idx = row*ncol + col;                 \n"
"   float imval = image[im_idx];                 \n"
"   float u = row-cenrow;                        \n" 
"   float v = col-cencol;                        \n"
"   float chi2=icc*u*u + irr*v*v - 2.0*irc*u*v;  \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"
"\n"
"   u = row-1.1*cenrow;                              \n" 
"   v = col-1.1*cencol;                              \n"
"   chi2=1.01*icc*u*u + 0.98*irr*v*v - .999*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"
"\n"
"   u = row-0.99*cenrow;                              \n" 
"   v = col-0.99*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-0.97*cenrow;                              \n" 
"   v = col-0.93*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-1.01*cenrow;                              \n" 
"   v = col-1.03*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-.9991*cenrow;                              \n" 
"   v = col-1.0009*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-.983*cenrow;                              \n" 
"   v = col-1.31*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-0.993*cenrow;                              \n" 
"   v = col-0.99999*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-1.11*cenrow;                              \n" 
"   v = col-1.14*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"

"\n"
"   u = row-.975*cenrow;                              \n" 
"   v = col-1.00*cencol;                              \n"
"   chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;    \n"
"   chi2 *= idet;                                \n"
"   tmp += exp( -0.5*chi2 );  \n"



"\n"
"   tmp = tmp-imval;             \n"
"   output[idx] = -0.5*tmp*tmp;  \n"

"}                                                                     \n";
//END KERNEL
const char *kernel_source_old =
"__kernel void simple(                                                   \n"
"   global float* output, int nel)                                       \n"
"{                                                                       \n"
"   int index = get_global_id(0);                                        \n"
"   if (index >= nel)                                                    \n"
"       return;                                                          \n"
"   output[index] = exp( 0.5*log((float)index));                         \n"
"}                                                                       \n";


#define NWALKERS 20
//#define NWALKERS 32
//#define NWALKERS 1

// Storage for the arrays.
static cl_mem output;
// OpenCL state
static cl_command_queue queue;
//static cl_kernel kernel;

// we expect this many
#define NDEV 4
static cl_device_id device_ids[NDEV];
static cl_context context;

static cl_platform_id platform_id;

static const cl_int nrow=25;
static const cl_int ncol=25;

void compare_data(int nwalkers, int nrow, int ncol, float *gpudata, float *cpudata)
{
    float maxdiff=0.;
    float diff=0;
    for (int iwalk=0; iwalk<nwalkers; iwalk++) {
        for (int row=0; row<nrow; row++) {
            for (int col=0; col<ncol; col++) {
                //printf("walk: %d row: %d col: %d diff: %g\n", iwalk,row,col,diff);
                int idx=iwalk*nrow*ncol + row*ncol + col;
                diff=fabs( gpudata[idx]-cpudata[idx]);
                if (diff > maxdiff) {
                    maxdiff=diff;
                }
            }
        }
    }
    printf("max diff: %.16g\n", maxdiff);
}
// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return (size_t)global_size;
    } else 
    {
        return (size_t)(global_size + group_size - r);
    }
}

void do_c_map(int iwalker,
              int nrow, 
              int ncol, 
              float cenrow,
              float cencol,
              float idet,
              float irr,
              float irc,
              float icc,
              float *data, 
              float *image)
{
    for (int row=0; row<nrow; row++) {
        for (int col=0; col<ncol; col++) {

            float imval=image[row*ncol + col];

            int idx=iwalker*nrow*ncol + row*ncol + col;

            float u = row-cenrow;
            float v = col-cencol;
            float chi2=icc*u*u + irr*v*v - 2.0*irc*u*v;
            float tmp=0;
            chi2 *= idet;
            tmp = expd( -0.5*chi2 );

            u = row-1.1*cenrow;
            v = col-1.1*cencol;
            chi2=1.01*icc*u*u + 0.98*irr*v*v - .999*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-0.99*cenrow;
            v = col-0.99*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-0.97*cenrow;
            v = col-0.93*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

 
            u = row-1.01*cenrow;
            v = col-1.03*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );


            u = row-0.9991*cenrow;
            v = col-1.0009*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-0.983*cenrow;
            v = col-1.31*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-0.993*cenrow;
            v = col-0.99999*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-1.11*cenrow;
            v = col-1.14*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            u = row-0.975*cenrow;
            v = col-1.00*cencol;
            chi2=1.1*icc*u*u + .979*irr*v*v - 1.001*2.0*irc*u*v;
            chi2 *= idet;
            tmp += expd( -0.5*chi2 );

            tmp = tmp-imval;
            data[idx] = -0.5*tmp*tmp;

        }
    }
}

//END PROFILING

int main(int argc, char** argv)
{

    if (argc < 4) {
        printf("%s nrepeat docpu device\n", argv[0]);
        exit(1);
    }

    int nrepeat=atoi(argv[1]);
    int docpu=atoi(argv[2]);
    int devnum=atoi(argv[3]);

    float cenrow0=12.;
    float cencol0=12.;
    float irr0=2.;
    float irc0=0.;
    float icc0=3.;

    int nelem=nrow*ncol;

    int ntot=nrow*ncol*NWALKERS;

    cl_uint numPlatforms;
    cl_int err = CL_SUCCESS;

    clock_t t0,t1;
    int nsteps=600;

    int device_type=0;
    if (1) {
        device_type=CL_DEVICE_TYPE_GPU;
    } else {
        device_type=CL_DEVICE_TYPE_CPU;
    }

    //SETUP PLATFORM
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not get platform\n");
        exit(EXIT_FAILURE);
    }
    if(numPlatforms > 0)
    {
        //we have at least one
        //cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        cl_platform_id* platforms = calloc(numPlatforms, sizeof(cl_platform_id));
        err = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr,"could not get platform id\n");
            exit(EXIT_FAILURE);
        }

        fprintf(stderr,"Found %d platforms\n", numPlatforms);
        platform_id = platforms[0];
        //delete[] platforms;
        free(platforms);
    }
    else
        exit(0);
    //END PLATFORM

    //SETUP CONTEXT
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform_id,
        0
    };

    context = clCreateContextFromType(
            cps,
            device_type,
            NULL,
            NULL,
            &err);
    //END CONTEXT

    int num_devices=0;
    err = clGetDeviceIDs(platform_id, device_type, NDEV, device_ids, &num_devices);
    fprintf(stderr,"found %d devices\n", num_devices);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not get device ids\n");
        exit(EXIT_FAILURE);
    }
    if (NDEV != num_devices) {
        printf("expected %d devices\n", NDEV);
        exit(1);
    }

    //int devnum=-1;
    for (int i=0; i<num_devices; i++) {
        size_t len=0;
        cl_uint avail=0;
        cl_uint id=0;
        
        clGetDeviceInfo(device_ids[i], CL_DEVICE_AVAILABLE, sizeof(cl_uint), &avail, &len);
        clGetDeviceInfo(device_ids[i], CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &id, &len);
        printf("device #: %d id: %d avail: %d\n", i, id, avail);
        /*
        if (avail && devnum==-1) {
            //devnum=i;
            //break;
        }
        */
    }
    /*
    if (devnum==-1) {
        printf("no available devices\n");
        exit(1);
    }
    */
    printf("choosing device %d\n", devnum);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source , NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create program\n");
        exit(EXIT_FAILURE);
    }




    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_ids[devnum], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

    clGetDeviceInfo(device_ids[devnum], CL_DEVICE_NAME, sizeof(buffer), buffer, &len);
    cl_ulong memsize;
    clGetDeviceInfo(device_ids[devnum], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, &len);
    cl_uint nunits;
    clGetDeviceInfo(device_ids[devnum], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, &len);
    cl_ulong max_work_group_size;
    clGetDeviceInfo(device_ids[devnum], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &max_work_group_size, &len);
    //cl_uint warp_size;
    //clGetDeviceInfo(device_ids[devnum], CL_NV_DEVICE_WARP_SIZE, sizeof(cl_uint), &warp_size, &len);
     
    printf("CL_DEVICE_NAME:                    '%s'\n", buffer);
    printf("CL_DEVICE_GLOBAL_MEM_SIZE:          %lu\n", memsize);
    // compute unit is a lump of hardware that executes 'work groups'
    printf("CL_DEVICE_MAX_COMPUTE_UNITS:        %u\n", nunits);
    // max number of items per work group
    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", max_work_group_size);
    //printf("CL_NV_DEVICE_WARP_SIZE:             %u\n", warp_size);

    size_t szLocalWorkSize = nrow;
    //size_t szLocalWorkSize = 512;
    // make sure multiple of 32
    szLocalWorkSize=shrRoundUp((int)256, (int)szLocalWorkSize);
    // rounded up to the nearest multiple of the LocalWorkSize
    size_t szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, (int)ntot);

    printf("nrow: %d\n", nrow);
    printf("ncol %d\n", ncol);
    printf("setting nelem: %d\n", nelem);
    printf("setting ntot: %d\n", ntot);
    printf("setting local work size: %lu\n", szLocalWorkSize);
    printf("setting global work size: %lu\n", szGlobalWorkSize);


    cl_float *data_from_gpu = calloc(szGlobalWorkSize, sizeof(cl_float));
    cl_float *data_from_cpu = calloc(szGlobalWorkSize, sizeof(cl_float));

    cl_float *rows=calloc(szGlobalWorkSize,sizeof(cl_float));
    cl_float *cols=calloc(szGlobalWorkSize,sizeof(cl_float));
    cl_float *image=calloc(nrow*ncol,sizeof(cl_float));

    for (int iwalk=0; iwalk<NWALKERS; iwalk++) {
        for (int row=0; row<nrow; row++) {
            for (int col=0; col<ncol; col++) {
                int idx = iwalk*(nrow*ncol) + row*ncol + col;
                rows[idx] = row;
                cols[idx] = col;
                image[row*ncol + col] = 3;
            }
        }
    }



    //queue = clCreateCommandQueue(context, device_ids, 0, &err);
    queue = clCreateCommandQueue(context, 
                                 device_ids[devnum],
                                 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 
                                 &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create command queue\n");
        exit(EXIT_FAILURE);
    }


    cl_mem rows_in = clCreateBuffer(context,  
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_float)*szGlobalWorkSize, rows, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create rows buffer\n");
        exit(EXIT_FAILURE);
    }
    cl_mem cols_in = clCreateBuffer(context,  
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_float)*szGlobalWorkSize, cols, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create cols buffer\n");
        exit(EXIT_FAILURE);
    }

    // we copy because we want the memory to be zerod
    // read-write because we will reduce it!
    output = clCreateBuffer(context,  
            CL_MEM_READ_WRITE,  sizeof(cl_float)*szGlobalWorkSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create buffer\n");
        exit(EXIT_FAILURE);
    }

    //OPTIMIZATION OPTIONS FOUND AT http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html

    err = clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not build program\n");
        exit(EXIT_FAILURE);
    }



    //SETUP KERNEL
    cl_kernel kernel = clCreateKernel(program, "gmix", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not create kernel\n");
        exit(EXIT_FAILURE);
    }


    clReleaseProgram(program); // no longer needed


    printf("processing %dx%d image %d walkers %d steps nrepeat %d\n",
           nrow,ncol,NWALKERS,nsteps,nrepeat);
    double tstandard=0;
    double topencl=0;

    err =  clSetKernelArg(kernel, 0, sizeof(cl_int), &ntot);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_int), &ncol);


    err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &rows_in);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not set rows arg\n");
        exit(EXIT_FAILURE);
    }
    err |= clSetKernelArg(kernel, 10, sizeof(cl_mem), &cols_in);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not set cols arg\n");
        exit(EXIT_FAILURE);
    }
    err |=  clSetKernelArg(kernel, 11, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not set output arg\n");
        exit(EXIT_FAILURE);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr,"could not set kernel args\n");
        exit(EXIT_FAILURE);
    }


    srand48(10);
    t0=clock();
    for (int rep=0; rep<nrepeat; rep++) {

        // each repeat represents pushing a new image in
        err=0;
        cl_mem image_in = clCreateBuffer(context,  
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_float)*nrow*ncol, image, &err);
        err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &image_in);
        if (err != CL_SUCCESS) {
            fprintf(stderr,"could not create image buffer\n");
            exit(EXIT_FAILURE);
        }

        for (int step=0; step<nsteps; step++) {

            float cenrow = cenrow0 + 0.01*(drand48()-0.5);
            float cencol = cencol0 + 0.01*(drand48()-0.5);
            float irr = irr0+0.01*(drand48()-0.5);
            float irc = irc0+0.01*(drand48()-0.5);
            float icc = icc0+0.01*(drand48()-0.5);
            float det = irr*icc - irc*irc;
            float idet = 1./det;

            // a copy of the kernel is made each time, so we can add new arguments
            err |=  clSetKernelArg(kernel, 2, sizeof(cl_float), (void*)&cenrow);
            err |=  clSetKernelArg(kernel, 3, sizeof(cl_float), (void*)&cencol);
            err |=  clSetKernelArg(kernel, 4, sizeof(cl_float), (void*)&idet);
            err |=  clSetKernelArg(kernel, 5, sizeof(cl_float), (void*)&irr);
            err |=  clSetKernelArg(kernel, 6, sizeof(cl_float), (void*)&irc);
            err |=  clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&icc);

            if (err != CL_SUCCESS) {
                fprintf(stderr,"could not set step kernel args\n");
                exit(EXIT_FAILURE);
            }

            err = clEnqueueNDRangeKernel(queue, 
                    kernel, 
                    1, 
                    NULL, 
                    &szGlobalWorkSize, 
                    &szLocalWorkSize, 
                    0, 
                    NULL, 
                    NULL);

            if (err != CL_SUCCESS) {
                fprintf(stderr,"error executing kernel\n");
                exit(EXIT_FAILURE);
            }

        }
        clReleaseMemObject(image_in);
    }
    t1=clock();
    topencl = ((double)(t1-t0))/CLOCKS_PER_SEC;


    printf("time for GPU: %lf\n", topencl);

    if (docpu) {
        printf("doing cpu\n");
        srand48(10);
        t0=clock();
        for (int rep=0; rep<nrepeat; rep++) {
            for (int step=0; step<nsteps; step++) {
                float cenrow = cenrow0 + 0.01*(drand48()-0.5);
                float cencol = cencol0 + 0.01*(drand48()-0.5);
                float irr = irr0+0.01*(drand48()-0.5);
                float irc = irc0+0.01*(drand48()-0.5);
                float icc = icc0+0.01*(drand48()-0.5);
                float det = irr*icc - irc*irc;
                float idet = 1./det;

                for (int iwalk=0; iwalk<NWALKERS; iwalk++) {
                    do_c_map(iwalk, nrow, ncol, cenrow, cencol, idet, irr, irc, icc, data_from_cpu,image);
                }
            }
        }
        t1=clock();
        tstandard += ((double)(t1-t0))/CLOCKS_PER_SEC;
        printf("time for C loop: %lf\n", tstandard);
        printf("opencl was %.16g times faster\n", tstandard/topencl);


        err = clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(cl_float)*szGlobalWorkSize, data_from_gpu, 0, 
                NULL, NULL );
        if (err != CL_SUCCESS) {
            fprintf(stderr,"error reading buffer into local\n");
            exit(EXIT_FAILURE);
        }


        compare_data(NWALKERS, nrow, ncol, data_from_gpu, data_from_cpu);
    }



    //BEGIN CLEANUP
    clReleaseMemObject(rows_in);
    clReleaseMemObject(cols_in);
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //END CLEANUP

    return 0;
}
