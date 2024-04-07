# Installation

## Docker 

1. Install Docker following the [official instructions](https://docs.docker.com/engine/install/)
2. Build the container: `docker build -t treeflow .`
3. Run TreeFlow
    * To run Jupyter Lab:
        1. `docker run treeflow -p 8888:8888`
            * To use a different port (e.g. 8999) use `docker run -p 8999:8888 treeflow` 
            * If you need to access a local directory (e.g. for input and output files) mount it into the docker image: `docker run treeflow -v ../repo/data:data` to mount the directory `../repo/data` to the `data` directory in the notebook
        2. Follow the link that appears to Jupyter (`http://127.0.0.1:8888/lab?token=...`), changing the port if necessary
        3. To stop the process, use `docker kill {container}` (you can use `docker ps` to lookup the ID)
            * On Linux `docker ps | grep treeflow | awk '{print $1}' | xargs docker kill` will stop all TreeFlow containers
    * To run one of [TreeFlow's command line applications](cli):
        * `docker run treeflow {command}`
        * You may want to mount a data directory for input/output e.g. `docker run -v ../repo/data:data treeflow_vi -i data/alignment.fasta -t data/topology.nwk --tree-samples-output data/tree-results.nexus`
        
  