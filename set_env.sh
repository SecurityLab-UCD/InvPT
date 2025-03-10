# Sets the environment variables in conda. This only needs to run once every
# update. Make sure the environment is running before the script is run.

# set_env <name> <value>
set_env() {
    name=$1
    value=$2
    conda env config vars set $name=$value > /dev/null
}
set_env 'PYTHONPATH'    "$PYTHONPATH:`pwd`"
set_env 'PIA_HOME'      "`pwd`"
set_env 'JDK_LIB'       "/usr/lib/jvm/java-11-openjdk-amd64/lib"
