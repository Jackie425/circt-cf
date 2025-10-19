/*
 * Simple Test Program for OR1200 - C Version
 * 
 * This program performs basic arithmetic and stores results to memory.
 * Stack is initialized by crt0.S startup code.
 */

int main(void) {
    // Use volatile to prevent optimization
    volatile int memory[6];
    
    // Use small values that fit in 16-bit signed range
    volatile int a = 0x1234;
    volatile int b = 0x4321;
    volatile int sum, diff, prod;
    
    // Test 1: Addition
    sum = a + b;
    memory[0] = sum;
    
    // Test 2: Subtraction
    diff = b - a;
    memory[1] = diff;
    
    // Test 3: Multiply by 2 (shift left)
    prod = a + a;
    memory[2] = prod;
    
    // Test 4: Store original values
    memory[3] = a;
    memory[4] = b;
    
    // Run for a fixed number of iterations instead of infinite loop
    volatile int counter = 0;
    for (int i = 0; i < 100; i++) {
        counter = counter + 1;
        memory[5] = counter;
        memory[6] = sum + counter;
    }
    
    // Return success
    return 0;
}
