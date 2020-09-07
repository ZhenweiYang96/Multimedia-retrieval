# Our MR pipeline

Lorem ipsum

## Running the pipeline

For our dependency management we utilize a tool called `pipenv`, this will create a virtualized python install with only the dependecies installed that are needed to run this program. To install pipenv you can lookup [their guide](https://pypi.org/project/pipenv/).
When you have pipenv running you can use the following shell commands to start the tool:

``` sh
# Install the deps
pipenv sync
# Run a python file
pipenv run ./project/flipping.py
# Or, enter a the virtual env and run it from there
pipenv shell
./project/flipping.py
```