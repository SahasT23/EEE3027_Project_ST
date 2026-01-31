#include <stdio.h>   // standard i/o libraries
#include <stdlib.h>  

// Define the FIFO buffer structure
typedef struct {
    int *data;       // Pointer to the array that holds the buffer data
    int head;        // Index where the next piece of data will be written
    int tail;        // Index where the next piece of data will be read from
    int count;       // How many items are currently in the buffer
    int size;        // The total number of slots in the buffer
} FIFO;

// Function to create a new FIFO buffer with a given number of slots
FIFO *fifo_create(int num_slots) {
    FIFO *fifo = malloc(sizeof(FIFO));  // Allocate memory for the FIFO structure
    fifo->data = malloc(sizeof(int) * num_slots);  // Allocate memory for the data array
    fifo->head = 0;    // Start writing at position 0
    fifo->tail = 0;    // Start reading at position 0
    fifo->count = 0;   // Buffer starts empty
    fifo->size = num_slots;  // Store the total number of slots
    return fifo;       // Return a pointer to the new FIFO
}

// Function to add a value to the buffer (enqueue)
int fifo_push(FIFO *fifo, int value) {
    if (fifo->count == fifo->size) {  // Check if the buffer is full
        printf("Buffer is full. Cannot push %d.\n", value);  // Print an error
        return -1;  // Return -1 to indicate failure
    }
    fifo->data[fifo->head] = value;  // Write the value at the head position
    fifo->head = (fifo->head + 1) % fifo->size;  // Move head forward, wrap around if needed
    fifo->count = fifo->count + 1;  // Increase the item count by one
    return 0;  // Return 0 to indicate success
}

// Function to remove a value from the buffer (dequeue)
int fifo_pop(FIFO *fifo, int *output) {
    if (fifo->count == 0) {  // Check if the buffer is empty
        printf("Buffer is empty. Cannot pop.\n");  // Print an error
        return -1;  // Return -1 to indicate failure
    }
    *output = fifo->data[fifo->tail];  // Read the value at the tail position
    fifo->tail = (fifo->tail + 1) % fifo->size;  // Move tail forward, wrap around if needed
    fifo->count = fifo->count - 1;  // Decrease the item count by one
    return 0;  // Return 0 to indicate success
}

// Function to free the buffer memory when done
void fifo_destroy(FIFO *fifo) {
    free(fifo->data);  // Free the data array
    free(fifo);         // Free the FIFO structure itself
}

// Main function to test the FIFO buffer
int main() {
    int value;  // Variable to store popped values

    // Create a buffer with 3 slots (dynamic size)
    FIFO *fifo = fifo_create(3);

    // Test 1: Push three items and pop them in order
    fifo_push(fifo, 10);
    fifo_push(fifo, 20);
    fifo_push(fifo, 30);

    fifo_pop(fifo, &value);
    printf("Expected 10, got %d\n", value);

    fifo_pop(fifo, &value);
    printf("Expected 20, got %d\n", value);

    fifo_pop(fifo, &value);
    printf("Expected 30, got %d\n", value);

    // Test 2: Push when full
    fifo_push(fifo, 1);
    fifo_push(fifo, 2);
    fifo_push(fifo, 3);
    fifo_push(fifo, 4);  // Should print "Buffer is full"

    // Test 3: Pop when empty
    fifo_pop(fifo, &value);
    fifo_pop(fifo, &value);
    fifo_pop(fifo, &value);
    fifo_pop(fifo, &value);  // Should print "Buffer is empty"

    // Test 4: Wraparound works
    fifo_push(fifo, 40);
    fifo_push(fifo, 50);
    fifo_pop(fifo, &value);
    printf("Expected 40, got %d\n", value);
    fifo_pop(fifo, &value);
    printf("Expected 50, got %d\n", value);

    fifo_destroy(fifo);  // Free all memory

    return 0;
}