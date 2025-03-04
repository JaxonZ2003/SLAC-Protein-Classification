import unittest
import os
import pandas as pd
import torch
from dataset import ImageDataset

class TestImageDataset(unittest.TestCase):

    def setUp(self):
        """Set up a dataset for testing"""
        self.testDataset = ImageDataset('../data/train_info.csv')
  
    def test_dataset_initialization(self):
        """Test if dataset initializes correctly"""
        self.assertIsNotNone(self.testDataset)

    def test_dataset_type(self):
        """Test dataframe structure and columns"""
        expected_columns = ['image_path', 'image_id', 'label_id', 'label_text', 'label_raw', 'source']
        self.assertIsInstance(self.testDataset.dataframe, pd.DataFrame)
        self.assertListEqual(self.testDataset.dataframe.columns.tolist(), expected_columns)
        self.assertEqual(self.testDataset.dataframe.shape[1], 6)
        self.assertFalse(self.testDataset.dataframe.empty)

    def test_dataframe_not_empty(self):
        """Test if dataframe contains data"""
        self.assertFalse(self.testDataset.dataframe.empty)

    def test_len_method(self):
        """Test dataset length matches dataframe rows"""
        self.assertEqual(len(self.testDataset), self.testDataset.dataframe.shape[0])

    def test_getitem_types(self):
        """Test __getitem__ returns correct types"""
        img, label = self.testDataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.dtype, torch.long)

    def test_getitem_label_correct(self):
        """Test labels match dataframe values"""
        index = 0
        expected_label = self.testDataset.dataframe.iloc[index]['label_id']
        _, actual_label = self.testDataset[index]
        self.assertEqual(actual_label.item(), expected_label)

    def test_labeldict_integrity(self):
        """
        Test that labeldict has the correct row indices for each label_id.
        That is, for each label_id, labeldict[label_id] should be the list of all
        dataframe indices whose 'label_id' column matches label_id.
        """
        # For every label_id in the dictionary, verify the stored list of indices is correct.
        for label_id, stored_indices in self.testDataset.labeldict.items():
            # The "expected" indices for this label_id:
            expected_indices = self.testDataset.dataframe.index[
                self.testDataset.dataframe['label_id'] == label_id
            ].tolist()
            
            self.assertListEqual(
                stored_indices,
                expected_indices,
                msg=f"label_id {label_id} does not match expected row indices"
            )

        # Also verify that every unique label_id from the DF is present in labeldict
        all_label_ids = self.testDataset.dataframe['label_id'].unique()
        for label_id in all_label_ids:
            self.assertIn(
                label_id,
                self.testDataset.labeldict,
                msg=f"label_id {label_id} not found in the labeldict"
            )

    def test_summary_properties(self):
        """Test dataset summary metrics"""
        self.assertGreater(self.testDataset.datasize, 0)
        self.assertIsInstance(self.testDataset.labeldict, dict)
        self.assertEqual(self.testDataset.numLabel, len(self.testDataset.labeldict))

    def test_image_tensor_shape(self):
        """Test image tensor dimensions"""
        img, _ = self.testDataset[0]
        self.assertEqual(img.ndim, 3)  # Should have 3 dimensions (C, H, W)
        self.assertEqual(img.shape[0], 3)  # Should have 3 color channels

    def test_image_files_exist(self):
        """Verify image files exist on disk"""
        # Test first 5 entries
        for idx in [0, 1, 2, 3, 4]:
            img_path = self.testDataset.dataframe.iloc[idx]['image_path']
            self.assertTrue(os.path.isfile(img_path), f"Missing image: {img_path}")

if __name__ == '__main__':
    unittest.main()