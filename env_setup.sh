work_dir=$(cd "$(dirname "$0")" && pwd)
echo "previous PYTHONPATH is $PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$work_dir/models:$work_dir/utils
echo "present PYTHONPATH is $PYTHONPATH"
