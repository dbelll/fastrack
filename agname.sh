#!/bin/bash
# Rename agent output files
#
# rename <num_agents> <num_opponents> <episode_length> <num_episodes>
#

if (test $# -lt 4)
then
    echo "Usage: rename <num_agents> <num_opponents> <episode_length> <num_episodes>"
fi

for n in 0 1 2 3
do
  eval  "_agfile=GPU${n}n4_$1v$2_$3x$4$5$6.agent"
#  echo "_agfile is $_agfile"
  if [ -f GPU$n.agent ]
      then
      if [ -f $_agfile ]
	  then
	  echo $_agfile already exists
      else
	  echo creating $_agfile
	  mv GPU$n.agent $_agfile
      fi
  else
      echo GPU$n.agent" not found!"
  fi
done

