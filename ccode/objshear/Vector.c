// This file was auto-generated

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include "Vector.h"


struct szvector* szvector_new(size_t num) {

    struct szvector* vector = malloc(sizeof(struct szvector));
    if (vector == NULL) {
        printf("Could not allocate struct szvector\n");
        exit(EXIT_FAILURE);
    }

    vector->size = num;

    if (num > 0) {
        vector->data = calloc(num, sizeof(size_t));
        if (vector->data == NULL) {
            printf("Could not allocate %ld size_t in vector\n", num);
            exit(EXIT_FAILURE);
        }
    } else {
        vector->data=NULL;
    }
    return vector;
}

struct szvector* szvector_fromarray(size_t* arr, size_t size) {

    struct szvector* vector = szvector_new(size);

    if (size > 0) {
        memcpy(vector->data, arr, size*sizeof(size_t));
    }
    return vector;
}

void szvector_resize(struct szvector* vector, size_t newsize) {
    if (vector==NULL) {
        printf("Attempt to resize unallocated size_t vector\n");
        exit(EXIT_FAILURE);
    }

    size_t oldsize=vector->size;
    size_t elsize = sizeof(size_t);

    size_t* newdata = realloc(vector->data, newsize*elsize);

    if (newdata == NULL) {
        printf("failed to reallocate\n");
        exit(EXIT_FAILURE);
    }

    if (newsize > oldsize) {
        // realloc does not zero the new memory
        memset(newdata+oldsize, 0, (newsize-oldsize)*elsize);
    }

    vector->data = newdata;
    vector->size = newsize;
}

void szvector_delete(struct szvector* vector) {
    if (vector != NULL) {
        free(vector->data);
        free(vector);
        vector=NULL;
    }
}




struct i64vector* i64vector_new(size_t num) {

    struct i64vector* vector = malloc(sizeof(struct i64vector));
    if (vector == NULL) {
        printf("Could not allocate struct i64vector\n");
        exit(EXIT_FAILURE);
    }

    vector->size = num;

    if (num > 0) {
        vector->data = calloc(num, sizeof(int64_t));
        if (vector->data == NULL) {
            printf("Could not allocate %ld int64_t in vector\n", num);
            exit(EXIT_FAILURE);
        }
    } else {
        vector->data=NULL;
    }
    return vector;
}

struct i64vector* i64vector_fromarray(int64_t* arr, size_t size) {

    struct i64vector* vector = i64vector_new(size);

    if (size > 0) {
        memcpy(vector->data, arr, size*sizeof(int64_t));
    }
    return vector;
}

void i64vector_resize(struct i64vector* vector, size_t newsize) {
    if (vector==NULL) {
        printf("Attempt to resize unallocated int64_t vector\n");
        exit(EXIT_FAILURE);
    }

    size_t oldsize=vector->size;
    size_t elsize = sizeof(int64_t);

    int64_t* newdata = realloc(vector->data, newsize*elsize);

    if (newdata == NULL) {
        printf("failed to reallocate\n");
        exit(EXIT_FAILURE);
    }

    if (newsize > oldsize) {
        // realloc does not zero the new memory
        memset(newdata+oldsize, 0, (newsize-oldsize)*elsize);
    }

    vector->data = newdata;
    vector->size = newsize;
}

void i64vector_delete(struct i64vector* vector) {
    if (vector != NULL) {
        free(vector->data);
        free(vector);
        vector=NULL;
    }
}




struct f64vector* f64vector_new(size_t num) {

    struct f64vector* vector = malloc(sizeof(struct f64vector));
    if (vector == NULL) {
        printf("Could not allocate struct f64vector\n");
        exit(EXIT_FAILURE);
    }

    vector->size = num;

    if (num > 0) {
        vector->data = calloc(num, sizeof(float64));
        if (vector->data == NULL) {
            printf("Could not allocate %ld float64 in vector\n", num);
            exit(EXIT_FAILURE);
        }
    } else {
        vector->data=NULL;
    }
    return vector;
}

struct f64vector* f64vector_fromarray(float64* arr, size_t size) {

    struct f64vector* vector = f64vector_new(size);

    if (size > 0) {
        memcpy(vector->data, arr, size*sizeof(float64));
    }
    return vector;
}

void f64vector_resize(struct f64vector* vector, size_t newsize) {
    if (vector==NULL) {
        printf("Attempt to resize unallocated float64 vector\n");
        exit(EXIT_FAILURE);
    }

    size_t oldsize=vector->size;
    size_t elsize = sizeof(float64);

    float64* newdata = realloc(vector->data, newsize*elsize);

    if (newdata == NULL) {
        printf("failed to reallocate\n");
        exit(EXIT_FAILURE);
    }

    if (newsize > oldsize) {
        // realloc does not zero the new memory
        memset(newdata+oldsize, 0, (newsize-oldsize)*elsize);
    }

    vector->data = newdata;
    vector->size = newsize;
}

void f64vector_delete(struct f64vector* vector) {
    if (vector != NULL) {
        free(vector->data);
        free(vector);
        vector=NULL;
    }
}


