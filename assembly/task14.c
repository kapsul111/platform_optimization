#include <stdio.h>

int sum(int *numbers, int size) {
    int result = 0;
    for (int i = 0; i < size; i++) {
        result += numbers[i];
    }
    return result;
}

int sum_asm(int *numbers, int size) {
    int result = 0;
    
    asm ("xor %%eax, %%eax\n\t"
    "test %%ecx, %%ecx\n\t"
    "jz end\n\t"
    "mov %1, %%rbx\n\t"

    "cmp $4, %%ecx\n\t"
    "jle loop_start\n\t"

    "loop_big:\n\t"
    "add (%%rbx), %%rax\n\t"
    "add 4(%%rbx), %%rax\n\t"
    "add 8(%%rbx), %%rax\n\t"
    "add 12(%%rbx), %%rax\n\t"
    "add $16, %%rbx\n\t"
    "sub $4, %%ecx\n\t"
    "cmp $4, %%ecx\n\t"
    "jg loop_big\n\t"

    "loop_start:\n\t"
    "add (%%rbx), %%rax\n\t"
    "add $4, %%rbx\n\t"
    "loop loop_start\n\t"
    "end: \n\t"

    : "=a" (result)
    : "D" (numbers), "c" (size)
    : "rbx", "memory");

    return result;
}

int main() {
    int numbers[10] = {1,2,3,4,5,6,7,8,9,10};

    int res = sum_asm(numbers, 10);
    int res2 = sum(numbers, 10);

    printf("Result: %d %d\n", res, res2);

    return 0;
}