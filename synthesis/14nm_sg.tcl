remove_design -all
#******************************************************************************
# Set the design paths
#******************************************************************************
set LIBS ./library

# # the module that's gonna run
# set top_module "mac_lane"

# Multi-core run
set_host_options -max_cores 8

# Define the libraries and search path
set search_path [concat $search_path ${LIBS}]
set target_library ${LIBS}/14nm_sg_345K.db
set synthetic_library dw_foundation.sldb
set link_library [concat "*" $target_library $synthetic_library]

define_design_lib 14nm_sg -path ./14nm_sg
set rtl_path "./top"
set netlist_path "./netlists"
set rpt_path "./reports"
set script_path "./script"
set search_path [concat $search_path $rtl_path $script_path]
set sdc_path "./sdc"
set ddc_path "./ddc"

# the period of the clk input, unit: ps
set period 1429	
set uncertainty 50
set rpt_file "${top_module}.rpt"
set netlist_file "${top_module}.v"
set power_default_toggle_rate 0.1

set rpt_path [concat $rpt_path/$top_module]
file mkdir $rpt_path
#******************************************************************************
# Read RTL files
#******************************************************************************

analyze -library 14nm_sg -format sv {patchifier.sv FIFO.sv ReLU.sv SiLU.sv softmax.sv layer_mean.sv ln_forward.sv sqrt_mul.sv update_output.sv L1.sv add.sv post_sparsity.sv stochastic_rounding.sv L2.sv adder_tree.sv loss.sv scalar.sv LFSR.sv mac_lane.sv min_pooling.sv shifter.sv transposer.sv dataflow.sv mask.sv mul.sv sparsity.sv update_mask.sv} 

elaborate $top_module -library 14nm_sg

#-parameters "N=4"
check_design

set_max_delay -to [all_outputs] $period

#commented due to error: no "object list"
#set_size_only

# Make the blocks instantiated from the same reference unique
uniquify

#******************************************************************************
# generate clocks
#******************************************************************************

set clock_name "clk"

create_clock -name $clock_name -period $period $clock_name
set_clock_uncertainty $uncertainty $clock_name
set_dont_touch_network $clock_name

#set_dont_touch_network rst
#set_input_delay 250 -clock clk -max [all_inputs]
#set_input_delay 250 -clock clk -min [all_inputs]

#******************************************************************************
# compile setup and compile
#******************************************************************************

set_max_leakage_power 0.000000; # Optimize for minimum leakage
set_wire_load_model -name "1K"
set_wire_load_mode top

#Add buffers on inputs passed directly to outputs, or outputs passed directly
#to inputs
set compile_fix_multiple_port_nets true;

compile -map_effort high
ungroup -all -flatten
compile
check_design
write_file -format verilog -output "$netlist_path/$netlist_file"

set filename [format "%s%s"  $top_module ".sdc"]
write_sdc "$sdc_path/$filename"

set filename [format "%s%s"  $top_module ".ddc"]
write -f ddc -hier -output "$ddc_path/$filename"
#******************************************************************************
# generate  reports
#******************************************************************************
source report.tcl

quit
