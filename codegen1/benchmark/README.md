# Multi-Turn Programming Benchmark


## Format

Each line is a problem expressed in JSON format, consisting of the following fields:

* `id`: Problem ID
* `name`: Problem name (cf. Appendix D)
* `description`: A short description of the problem
* `category`: Manually labeled problem category
* `prompts`: A list of template-enabled strings, specifying each step.
* `inputs`: A list consisting of 5 test case inputs. Each test case is a key-value table mapping the variables (used in the templated prompt) to actual values.
* `outputs`: A list consisting of 5 test case outputs. Each test case is an expected output value of the program.
* `max_gen_length`: Maximum number of tokens we set for each turn for the problem. The value is  mostly 128 because each turn doesn't require substantial lines of code, but we adjusted a higher number when long generation is expected.
