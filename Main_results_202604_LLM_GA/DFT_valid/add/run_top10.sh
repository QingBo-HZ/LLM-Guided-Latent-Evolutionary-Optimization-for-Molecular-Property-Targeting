#!/bin/bash


MAX_JOBS=3
count=0

for f in *.gjf
do
    name=$(basename "$f" .gjf)

    echo "Running $name ..."
    g16 "$f" > "${name}.log" &

    ((count++))

    if [ $count -ge $MAX_JOBS ]; then
        wait
        count=0
    fi
done

wait
echo "All jobs finished."
