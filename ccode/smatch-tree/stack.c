// This file was auto-generated

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include "stack.h"


struct f64stack* f64stack_new(size_t num) {
    struct f64stack* stack = malloc(sizeof(struct f64stack));
    if (stack == NULL) {
        printf("Could not allocate struct f64stack\n");
        exit(EXIT_FAILURE);
    }

    stack->size = 0;
    stack->allocated_size = num;
    stack->push_realloc_style = STACK_PUSH_REALLOC_MULT;
    stack->push_initsize = STACK_PUSH_INITSIZE;
    stack->realloc_multval = STACK_PUSH_REALLOC_MULTVAL;

    if (num == 0) {
        stack->data = NULL;
    } else {
        stack->data = calloc(num, sizeof(float64));
        if (stack->data == NULL) {
            printf("Could not allocate data in pixlist\n");
            exit(EXIT_FAILURE);
        }
    }

    return stack;
}

void f64stack_realloc(struct f64stack* stack, size_t newsize) {

    size_t oldsize = stack->allocated_size;
    if (newsize != oldsize) {
        size_t elsize = sizeof(float64);

        float64* newdata = realloc(stack->data, newsize*elsize);
        if (newdata == NULL) {
            printf("failed to reallocate\n");
            exit(EXIT_FAILURE);
        }

        if (newsize > stack->allocated_size) {
            // the allocated size is larger.  make sure to initialize the new
            // memory region.  This is the area starting from index [oldsize]
            size_t num_new_bytes = (newsize-oldsize)*elsize;
            memset(&newdata[oldsize], 0, num_new_bytes);
        } else if (stack->size > newsize) {
            // The viewed size is larger than the allocated size in this case,
            // we must set the size to the maximum it can be, which is the
            // allocated size
            stack->size = newsize;
        }

        stack->data = newdata;
        stack->allocated_size = newsize;
    }

}
void f64stack_resize(struct f64stack* stack, size_t newsize) {
   if (newsize > stack->allocated_size) {
       f64stack_realloc(stack, newsize);
   }

   stack->size = newsize;
}

void f64stack_clear(struct f64stack* stack) {
    stack->size=0;
    stack->allocated_size=0;
    free(stack->data);
    stack->data=NULL;
}

struct f64stack* f64stack_delete(struct f64stack* stack) {
    if (stack != NULL) {
        f64stack_clear(stack);
        free(stack);
    }
    return NULL;
}

void f64stack_push(struct f64stack* stack, float64 val) {
    // see if we have already filled the available data vector
    // if so, reallocate to larger storage
    if (stack->size == stack->allocated_size) {

        size_t newsize;
        if (stack->allocated_size == 0) {
            newsize=stack->push_initsize;
        } else {
            // currenly we always use the multiplier reallocation  method.
            if (stack->push_realloc_style != STACK_PUSH_REALLOC_MULT) {
                printf("Currently only support push realloc style STACK_PUSH_REALLOC_MULT\n");
                exit(EXIT_FAILURE);
            }
            // this will "floor" the size
            newsize = (size_t)(stack->allocated_size*stack->realloc_multval);
            // we want ceiling
            newsize++;
        }

        f64stack_realloc(stack, newsize);

    }

    stack->size++;
    stack->data[stack->size-1] = val;
}

float64 f64stack_pop(struct f64stack* stack) {
    if (stack->size == 0) {
        return DBL_MAX;
    }

    float64 val=stack->data[stack->size-1];
    stack->size--;
    return val;
        
}

int __f64stack_compare_el(const void *a, const void *b) {
    double temp = 
        (  (double) *( (float64*)a ) ) 
         -
        (  (double) *( (float64*)b ) );
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}


void f64stack_sort(struct f64stack* stack) {
    qsort(stack->data, stack->size, sizeof(float64), __f64stack_compare_el);
}
float64* f64stack_find(struct f64stack* stack, float64 el) {
    return (float64*) bsearch(&el, stack->data, stack->size, sizeof(float64), __f64stack_compare_el);
}

struct i64stack* i64stack_new(size_t num) {
    struct i64stack* stack = malloc(sizeof(struct i64stack));
    if (stack == NULL) {
        printf("Could not allocate struct i64stack\n");
        exit(EXIT_FAILURE);
    }

    stack->size = 0;
    stack->allocated_size = num;
    stack->push_realloc_style = STACK_PUSH_REALLOC_MULT;
    stack->push_initsize = STACK_PUSH_INITSIZE;
    stack->realloc_multval = STACK_PUSH_REALLOC_MULTVAL;

    if (num == 0) {
        stack->data = NULL;
    } else {
        stack->data = calloc(num, sizeof(int64_t));
        if (stack->data == NULL) {
            printf("Could not allocate data in pixlist\n");
            exit(EXIT_FAILURE);
        }
    }

    return stack;
}

void i64stack_realloc(struct i64stack* stack, size_t newsize) {

    size_t oldsize = stack->allocated_size;
    if (newsize != oldsize) {
        size_t elsize = sizeof(int64_t);

        int64_t* newdata = realloc(stack->data, newsize*elsize);
        if (newdata == NULL) {
            printf("failed to reallocate\n");
            exit(EXIT_FAILURE);
        }

        if (newsize > stack->allocated_size) {
            // the allocated size is larger.  make sure to initialize the new
            // memory region.  This is the area starting from index [oldsize]
            size_t num_new_bytes = (newsize-oldsize)*elsize;
            memset(&newdata[oldsize], 0, num_new_bytes);
        } else if (stack->size > newsize) {
            // The viewed size is larger than the allocated size in this case,
            // we must set the size to the maximum it can be, which is the
            // allocated size
            stack->size = newsize;
        }

        stack->data = newdata;
        stack->allocated_size = newsize;
    }

}
void i64stack_resize(struct i64stack* stack, size_t newsize) {
   if (newsize > stack->allocated_size) {
       i64stack_realloc(stack, newsize);
   }

   stack->size = newsize;
}

void i64stack_clear(struct i64stack* stack) {
    stack->size=0;
    stack->allocated_size=0;
    free(stack->data);
    stack->data=NULL;
}

struct i64stack* i64stack_delete(struct i64stack* stack) {
    if (stack != NULL) {
        i64stack_clear(stack);
        free(stack);
    }
    return NULL;
}

void i64stack_push(struct i64stack* stack, int64_t val) {
    // see if we have already filled the available data vector
    // if so, reallocate to larger storage
    if (stack->size == stack->allocated_size) {

        size_t newsize;
        if (stack->allocated_size == 0) {
            newsize=stack->push_initsize;
        } else {
            // currenly we always use the multiplier reallocation  method.
            if (stack->push_realloc_style != STACK_PUSH_REALLOC_MULT) {
                printf("Currently only support push realloc style STACK_PUSH_REALLOC_MULT\n");
                exit(EXIT_FAILURE);
            }
            // this will "floor" the size
            newsize = (size_t)(stack->allocated_size*stack->realloc_multval);
            // we want ceiling
            newsize++;
        }

        i64stack_realloc(stack, newsize);

    }

    stack->size++;
    stack->data[stack->size-1] = val;
}

int64_t i64stack_pop(struct i64stack* stack) {
    if (stack->size == 0) {
        return INT64_MAX;
    }

    int64_t val=stack->data[stack->size-1];
    stack->size--;
    return val;
        
}

int __i64stack_compare_el(const void *a, const void *b) {
    int64_t temp = 
        (  (int64_t) *( (int64_t*)a ) ) 
         -
        (  (int64_t) *( (int64_t*)b ) );
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}


void i64stack_sort(struct i64stack* stack) {
    qsort(stack->data, stack->size, sizeof(int64_t), __i64stack_compare_el);
}
int64_t* i64stack_find(struct i64stack* stack, int64_t el) {
    return (int64_t*) bsearch(&el, stack->data, stack->size, sizeof(int64_t), __i64stack_compare_el);
}

// num is *allocated* size*; the stack will still show
// ->size==0 until you push
struct szstack* szstack_new(size_t num) {
    struct szstack* stack = malloc(sizeof(struct szstack));
    if (stack == NULL) {
        printf("Could not allocate struct szstack\n");
        exit(EXIT_FAILURE);
    }

    stack->size = 0;
    stack->allocated_size = num;
    stack->push_realloc_style = STACK_PUSH_REALLOC_MULT;
    stack->push_initsize = STACK_PUSH_INITSIZE;
    stack->realloc_multval = STACK_PUSH_REALLOC_MULTVAL;

    if (num == 0) {
        stack->data = NULL;
    } else {
        stack->data = calloc(num, sizeof(size_t));
        if (stack->data == NULL) {
            printf("Could not allocate data in pixlist\n");
            exit(EXIT_FAILURE);
        }
    }

    return stack;
}

void szstack_realloc(struct szstack* stack, size_t newsize) {

    size_t oldsize = stack->allocated_size;
    if (newsize != oldsize) {
        size_t elsize = sizeof(size_t);

        size_t* newdata = realloc(stack->data, newsize*elsize);
        if (newdata == NULL) {
            printf("failed to reallocate\n");
            exit(EXIT_FAILURE);
        }

        if (newsize > stack->allocated_size) {
            // the allocated size is larger.  make sure to initialize the new
            // memory region.  This is the area starting from index [oldsize]
            size_t num_new_bytes = (newsize-oldsize)*elsize;
            memset(&newdata[oldsize], 0, num_new_bytes);
        } else if (stack->size > newsize) {
            // The viewed size is larger than the allocated size in this case,
            // we must set the size to the maximum it can be, which is the
            // allocated size
            stack->size = newsize;
        }

        stack->data = newdata;
        stack->allocated_size = newsize;
    }

}
void szstack_resize(struct szstack* stack, size_t newsize) {
   if (newsize > stack->allocated_size) {
       szstack_realloc(stack, newsize);
   }

   stack->size = newsize;
}

void szstack_clear(struct szstack* stack) {
    stack->size=0;
    stack->allocated_size=0;
    free(stack->data);
    stack->data=NULL;
}

struct szstack* szstack_delete(struct szstack* stack) {
    if (stack != NULL) {
        szstack_clear(stack);
        free(stack);
    }
    return NULL;
}

void szstack_push(struct szstack* stack, size_t val) {
    // see if we have already filled the available data vector
    // if so, reallocate to larger storage
    if (stack->size == stack->allocated_size) {

        size_t newsize;
        if (stack->allocated_size == 0) {
            newsize=stack->push_initsize;
        } else {
            // currenly we always use the multiplier reallocation  method.
            if (stack->push_realloc_style != STACK_PUSH_REALLOC_MULT) {
                printf("Currently only support push realloc style STACK_PUSH_REALLOC_MULT\n");
                exit(EXIT_FAILURE);
            }
            // this will "floor" the size
            newsize = (size_t)(stack->allocated_size*stack->realloc_multval);
            // we want ceiling
            newsize++;
        }

        szstack_realloc(stack, newsize);

    }

    stack->size++;
    stack->data[stack->size-1] = val;
}

size_t szstack_pop(struct szstack* stack) {
    if (stack->size == 0) {
        return SIZE_MAX;
    }

    size_t val=stack->data[stack->size-1];
    stack->size--;
    return val;
        
}

int __szstack_compare_el(const void *a, const void *b) {
    int64_t temp = 
        (  (int64_t) *( (size_t*)a ) ) 
         -
        (  (int64_t) *( (size_t*)b ) );
    if (temp > 0)
        return 1;
    else if (temp < 0)
        return -1;
    else
        return 0;
}


void szstack_sort(struct szstack* stack) {
    qsort(stack->data, stack->size, sizeof(size_t), __szstack_compare_el);
}
size_t* szstack_find(struct szstack* stack, size_t el) {
    return (size_t*) bsearch(&el, stack->data, stack->size, sizeof(size_t), __szstack_compare_el);
}
