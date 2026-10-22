# create the sckeleton for aerial and ground data loader by planning
# it should inheritate from BaseStereoViewDataset like both MegaDepthAerial and MASt3RBaseStereoViewDataset
# it should implement dataloader and __getitem__
# it should manage data augmentation 
# it should handle resolution and required crops correctly (not like MASt3RBaseStereoViewDataset)
# it should provide a example at the end to load randomly data and visualize it 
# it should be usable by ign_aerial_ground_train (see /home/BSoheilian/work/dev/mast3r_ign/demo_duster_viz.py for visualization example)

