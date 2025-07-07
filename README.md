## How to Use This Model and Provide Inputs
Your goal is to create a new compression_... function that results in an Information Loss of 0.0.

Run the Code: Execute the script with the different example algorithms to see how they perform. Notice how compression_perfect_encoding gives zero loss, while the others give very large numbers.

Create Your Own Algorithm: Write a new Python function that takes r and M as input and returns g_tt and g_rr.

The Challenge: Your function must correctly calculate the Schwarzschild radius (rs) from the input mass M and then use it to define the metric components. The key is discovering the correct inverse relationship: g_tt = -(1 - rs/r) and g_rr = 1 / (1 - rs/r).
