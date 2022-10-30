treeflow_vi -s 1 \
    -i demo-data/h3n2.fasta \
    -m h3n2-model.yaml \
    -t demo-data/h3n2.nwk \
    -n 30000 \
    --learning-rate 0.001 \
    --init-values "clock_rate=0.003" \
    --trace-output demo-out/h3n2-trace.pickle \
    --samples-output demo-out/h3n2-samples.csv \
    --tree-samples-output demo-out/h3n2-trees.nexus \
    --n-output-samples 1000