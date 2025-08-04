# Lab2_AI_Team_23CLC03
Please install the libraries if this is the first time you are running the code:
pip install -r requirements.txt

Afterwards, head to the main function and run, there will be a prompt:
    Choose mode:
    1. Run all solvers on multiple inputs
    2. Choose a specific input and solver
Selecting 1 will run all 4 methods on examples given in Inputs: from input-01.txt to input-a.txt (with a is the selected number),
the results are shown in Outputs/output.txt, check the terminal message for time.

Selecting 2 will then also prompt you to select a number from 1 to 10 to run an example
(min 3x3 to max 25x25), after which there is another prompt:
    Choose a solving method:
    1. PySAT
    2. Brute-Force
    3. Backtrack
    4. A*
Select a method (PySAT recommended for inputs larger than input-04.txt), after which the
result will be shown in output-num.txt (input-10.txt will be saved to output-10.txt)