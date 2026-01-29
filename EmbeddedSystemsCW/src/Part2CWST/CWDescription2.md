## Coursework Part 2

  **Task:** a. Refactor the given sequential ML pipeline by decoupling key operations into self-contained classes.	
  
  **Worth:** 15%	
  
  **Task Breakdown:** The functions inside the given ML pipeline is successfully decoupled into several self-contained classes that can be instantiated and can still work when replacing the original function calls with the newly instantiated object calls. 

  ---

  **Task:** b. Integrate your shared FIFO buffer implementation from task 1b to the pipeline and ensure it works.	
  
  **Worth:** 15%	
  
  **Task Breakdown:** The (shared) FIFO buffer is successfully integrated into the ML pipeline and can hold the data used within the ML pipeline without greatly affecting its original performance.

  --- 

  **Task:** c. Convert the ML pipeline + FIFO buffer from task 2b to a concurrent solution, and analyse its performance.	
  
  **Worth:** 20%	
  
  **Task Breakdown:** The (inherently sequential) ML pipeline and its integrated shared FIFO buffer is now fully concurrent and managed by the mutex protocols, where its performance should (theoretically) be faster than the original sequential ML pipeline. 

  ---