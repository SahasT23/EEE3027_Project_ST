# Define functions
process_model()
 {
  xterm -e "echo $1 ; echo press Ctl-c to finish ; sleep 1h"
  return 0
 }

# Start process_1 first
process_model process_1

# Once it has finished start 
# process_2 and process_3 concurrently
process_model process_2 &
pid_process_2=$!
process_model process_3

# After both processes are finished start process_4
wait $pid_process_2
process_model process_4


