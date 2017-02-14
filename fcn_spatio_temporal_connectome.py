#!/usr/bin/python

import argparse
import sys
import os
import networkx as nx
import scipy.io
import numpy as np
import sklearn as sk
from sklearn import preprocessing



DESCRIPTION = """

This function outputs a spatio-temporal given a set of functional time series
and a structural connectivity matrix.

 INPUTs

 - SC_filename: csv file containing the subject structural
                connectivity matrix. Note that the structural connectivity
                matrix is assumed to be binary
                [string]
 - TS_filename: csv file contianing the matrix of the functional
                the time series (each row corresopnds to a single region
                time series). Note that the script expects the number of
                rows of the time series matrix to be equal to the dimension
                of the (square) structural conenctivity matrix 
                [string]
 - pos_threshold: threshold for point-process analysis
                  [float]
 - LABEL_filename = txt file containing the labels or identifiers of the
                    regions corresponding to the structural connectivity 
                    matrix and funcitonal time series. 
                    Each row of the file contains a single region identifier.
                    Note that the script expects the number of rows of the
                    LABEL_filename to be equal to the dimension of the (square)
                    structural connectivity matrix
 - output_prefix: prefix for the output files, e.g., '/output_dir/subjID_'
                  [string]


 OUTPUTs
 - z-scored time series (2D matrix, saved as mat file)
   ["output_prefix_"ts_zscore.mat"]
 - spatio-temporal connectome (gpickle format)
   ["output_prefix_"spatio-temporal_connectome.gpickle"]
 - connected components of the spaito-temporal connectome (Matlab structure, saved as mat file)
   ["output_prefix_"CC.mat"]
 - feature vectors (2D matrix, saved as mat file)
   ["output_prefix_"FM.mat"]


 REQUIRE
 	Python networkx
 	Python numpy
 	Python scipy
 	Python ssklearn
 	Python sys


 CREDITS	Alessandra Griffa
			alessandra.griffa@epfl.ch

			Department of Radiology
			Center hospitalier universitaire vaudois (CHUV) and Université de Lausanne (UNIL)
			Lausanne, Switzerland

			Signal Processing Laboratory LTS5
			École polytechnique fédérale de Lausanne (EPFL)
			Lausanne, Switzerland

			A. Griffa, B. Ricaud, K. Benzi, X. Bresson, A. Daducci, P. Vandergheynst, J.P. Thiran, P Hagmann (2017)
			Transient Networks of Spatio-temporal Connectivity Map Communication Pathways in Brain Functional Systems.
			Submitted to NeuroImage

"""



def buildArgsParser():
	p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
								description=DESCRIPTION)
	
	p.add_argument('SC_filename', action='store', metavar='SC_FILENAME', type=str,
					help='Binary structural connectivity matrix - csv file.')
	p.add_argument('TS_filename', action='store', metavar='TS_FILENAME', type=str,
					help='Functional time series matrix - csv file.')
	p.add_argument('pos_threshold', action='store', metavar='POSITIVE_THRESHOLD', type=str,
					help='Positive threshold value - number.')
	p.add_argument('LABEL_filename', action='store', metavar='LABEL_FILENAME', type=str,
					help='Binary structural connectivity matrix - txt file.')
	
	p.add_argument('output_prefix', action='store', metavar='OUTPUT_PREFIX', type=str,
					help='output prefix - e.g., ""output_directory/code_""')                 
	
	return p



def main():
	
	# Handel inputs
	# ======================================================================
	parser = buildArgsParser()
	args = parser.parse_args()

	if not os.path.isfile(args.SC_filename):
		parser.error('"{0}" must be a file!'.format(args.SC_filename))
	if not os.path.isfile(args.TS_filename):
		parser.error('"{0}" must be a file!'.format(args.TS_filename))
	if not os.path.isfile(args.LABEL_filename):
		parser.error('"{0}" must be a file!'.format(args.LABEL_filename))

	SC_filename = args.SC_filename
	TS_filename = args.TS_filename
	LABEL_filename = args.LABEL_filename
	pos_threshold = float(args.pos_threshold)
	output_dir = args.output_dir
	
	# Load inputs
	# [1] subject
	# [2] SC
	SC = np.loadtxt(open(SC_filename,"rb"), dtype=int, delimiter=',', skiprows=0)
	# Remove diagonal elements (not meaningful for the contruction of a spatio-temporal connectome)
	SC = SC - np.diag(np.diag(SC))
	# [3] ts
	ts = np.loadtxt(open(TS_filename, 'rb'), delimiter=',', skiprows=0)
	# [4] positive_threhsold value
	if pos_threshold <= 0:
		print >> sys.stderr, "A positive threshold value is expected!"
		sys.exit(2)
	# [5] ROI labels
	labels = open(LABEL_filename,'r').read().split('\n')
	if not labels[-1]:
		labels = labels[0:-1]
	# [6] output directory
	
	# Number of ROIs in the structural connectivity graph
	nROIs = ts.shape[0]
	# Number of time points
	ntp = ts.shape[1]
	# Sanity check
	if nROIs != SC.shape[0]:
		print >> sys.stderr, "The dimensions of SC and ts are not consistent!"
		sys.exit(3)
	
	
	
	# Define output file names
	# ======================================================================
	TS_zscore_filename = os.path.join(output_prefix + "_ts_zscore.mat")
	G_filename = os.path.join(output_prefix + "_spatio-temporal_connectome.gpickle")
	CC_filename = os.path.join(output_prefix + "_CC.mat")
	FM_filename = os.path.join(output_prefix + "_FM.mat")
	
	
	
	# If ts contains time series (not point processes), z-score ROI-wise time series
	# ======================================================================
	print('..... z-score time series .....')
	test = np.where(ts==0)
	test = np.array(test[0])
	testn = test.size
	test = np.where(ts==1)
	test = np.array(test[0])
	testn = testn + test.size
	if ts.size != testn:
		for i in xrange(len(ts)):
			ts[i] = sk.preprocessing.scale(ts[i], with_mean=True, with_std=True, copy=True)
		# Save z-scored time series
		print('..... write file: ' + TS_zscore_filename)
		scipy.io.savemat(TS_zscore_filename, dict(ts=ts))
	else:
		pos_threshold = 1
	
	
	
	# Create static structural connectivity graph Gs from the input adjacency matrix SC
	# ======================================================================
	# Create the extended multilayer graph from the adjacency matrix
	Gs = nx.from_numpy_matrix(SC)
	
	
	
	# Build a spatio-temporal connectome from SC and ts information
	# ======================================================================
	print('..... build multilayer network (spatio-temporal connectome) .....')
	# Initiate a new NetworkX graph
	G = nx.Graph()
	graph_data = {}
	graph_data['subject'] = output_prefix				# subject ID / output_prefix
	graph_data['ntp'] = ntp								# ntp, number of time points
	graph_data['nROIs'] = nROIs							# nROIs, number of anatomical ROIs
	graph_data['activation_threshold'] = pos_threshold	# nROIs, number of anatomical ROIs
	G.graph['graph_data'] = graph_data
	
	# Loop throughout all the time points (1 -> ntp-1)
	for t in range(ntp):
		
		# For current time point, find active ROIs
		tsst = ts[:,t] # fMRI values for all the nodes, current time point (t)
		active_nodes = np.where(tsst >= pos_threshold)[0]
		
		# Loop throughout all the ROIs active at current time point
		for i in active_nodes:
			
			# Generate a new node ID (in the multilayer network)
			# NOTE: each node in the multilayer network has a unique ID equal to layer_pos * nROIs + i, 
			#		with i anatomical_id of the considered node (from 1 to nROIs), layer_pos node position in time (from 1 to ntp),
			#		and nROIs number of ROIs in the structural connectivity graph
			node_id = t * nROIs + i + 1 # ROI IDs start from 1
			# If node_id does not exist in G, add it
			if ~G.has_node(node_id):
				# Generate attributes for the new node in the multilayer network
				node_attrs = {}
				node_attrs['anatomical_id'] = i + 1 # ROIs ID start from 1
				node_attrs['weight'] = tsst[i]
				node_attrs['tp'] = t + 1 # tp ID start from 1
				node_attrs['node_id'] = node_id
				node_attrs['label'] = labels[i]
				G.add_node(node_id, node_attrs)
			
			if t < (ntp-1):
				# Extract the neighbors of anatomical_id in Gs
				neighbor_nodes = Gs.neighbors(i)
				# Consider as well the node itself in the following (t+1) time point
				neighbor_nodes.extend([i])
				
				# Extract brain regions that are active at the following (t+1) time point
				tsstt = ts[:,t+1] # fMRI values for all the nodes, following (t+1) time point
				active_nodes_tt = np.where(tsstt >= pos_threshold)[0]
				
				# Intersect current region's neighbors, and regions active at the following (t+1) time point
				new_nodes = np.intersect1d(neighbor_nodes, active_nodes_tt, assume_unique=True)
				
				# Loop throughout all neighbor and active regions: add them to the multilayer network, together with the corresponding edges
				for j in new_nodes:
					node_id_new = ( (t+1) * nROIs ) + j + 1
					node_attrs = {}
					node_attrs['anatomical_id'] = j + 1
					node_attrs['weight'] = tsstt[j]
					node_attrs['tp'] = t + 2
					node_attrs['node_id'] = node_id_new
					node_attrs['label'] = labels[j]
					G.add_node(node_id_new, node_attrs)
					G.add_edge(node_id, node_id_new)
	
	# Add bi-directional links between nodes belonging to the same layer of Gs
	# Loop throughout all the time points (1 -> ntp)
	for t in range(ntp):
		
		# For current time point, find active nodes
		tsst = ts[:,t] # fMRI values for all the nodes, current time point
		active_nodes = np.where(tsst >= pos_threshold)[0]
		
		# Add links between active nodes at current time point, which are also neighbors in Gs
		for i in active_nodes:
			
			# Node ID in multilayer network
			node_id = t * nROIs + i + 1
			
			# Extract region neighbors in Gs
			neighbor_nodes = Gs.neighbors(i)
			
			# Intersect current node neighbors, and nodes active at the current (t) time point
			new_nodes = np.intersect1d(neighbor_nodes, active_nodes, assume_unique=True)
			
			# Add edges
			for j in new_nodes:
				node_id_new = t * nROIs + j + 1
				if ~G.has_edge(node_id, node_id_new):
					G.add_edge(node_id, node_id_new)
	
	# Save spatio-temporal connectome (multilayer network) as gpickle file
	print('..... write file: ' + G_filename)
	nx.write_gpickle(G, G_filename)
	
	
	
	# Extract the connected components (CCs) of the multilayer network
	# ======================================================================
	print('..... extract connected components (CCs) of multilayer network .....')
	# The output nx.connected_component_subgraphs() is a list of nx graphs, 
	# ordered from the largest to the smallest (in terms of number of nodes)
	CC = list(nx.connected_component_subgraphs(G))
	print('.....    (number of CCs: ' + str(len(CC)) + ')')
	
	# Set attributes of CCs: width (temporal extension), height (spatial extension), subject ID
	# And save the CCs to a Matlab array of structures
	height = np.zeros(shape=(len(CC)))
	width = np.zeros(shape=(len(CC)))
	subjectCC = []
	
	# Initialize Python list of dictionaries and feature matrix
	dictlist_cc = [dict() for x in range(len(CC))]
	FM = np.zeros(shape=(len(CC),nROIs))
	
	# Loop throughout all the connected components
	for i in range(0, len(CC)):
		
		# Current connected component
		cc = CC[i]
		# Nodes anatomical_id and layer_pos
		nodes = cc.nodes()
		
		anatomical_id_dict = nx.get_node_attributes(cc, 'anatomical_id')
		anatomical_id = [anatomical_id_dict[x] for x in nodes]
		
		tp_dict = nx.get_node_attributes(cc,'tp')
		tp = [tp_dict[x] for x in nodes]
		
		# Spatial height and temporal width of current component
		height[i] = len(set(anatomical_id))
		width[i] = max(tp) - min(tp) + 1
		
		# Feature vector
		ids = np.array(anatomical_id) - 1
		for j in range(0,len(ids)):
			FM[i][ids[j]] += 1
		
		# Fill-in Python dictionary
		dictlist_cc[i] = {'height':height[i],'width':width[i],'nodes':nodes,'edges':cc.edges(),'anatomical_id':anatomical_id,'tp':tp}
	
	# Save connected components list of dictionaries to Matlab format
	print('..... write file: ' + CC_filename)
	mdict = {'CC':dictlist_cc}
	scipy.io.savemat(CC_filename, mdict)
	
	# Normalize and save feature matrix
	print('..... write file: ' + FM_filename)
	FM_norms = np.apply_along_axis(np.linalg.norm, 1, FM)
	FM = FM.astype(float) / FM_norms[:, np.newaxis]
	scipy.io.savemat(FM_filename, mdict={'FM':FM})



if __name__ == "__main__":
    main()













