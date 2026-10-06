"""验证关键公式、信任边界、图连接与数据泄漏防护。"""

import base64
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from imos.agents import consistency, ActionPlanner
from imos.constraints import mismatch
from imos.data import load_dataset, perturb
from imos.graph import build_graph, BipartiteRegressor
from imos.io import read_json
from imos.metrics import performance, similarity, summarize
from imos.models import skeleton
from imos.trust import seal, trust_values, safe_path


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(Path("configs/experiment.json"))

    def test_candidate_count_and_unique_ids(self):
        models=skeleton()
        self.assertEqual(len(models),36)
        self.assertEqual(len({m["id"] for m in models}),36)

    def test_three_instruction_conflicts(self):
        self.assertFalse(consistency("同一轴承同时指定A型和B型")["accepted"])
        self.assertFalse(consistency("数据仅在本地处理并使用云端")["accepted"])
        self.assertFalse(consistency("使用振动数据预测但禁止读取")["accepted"])
        self.assertTrue(consistency("预测B型轴承粗糙度")["accepted"])

    def test_metrics_are_not_artificially_bounded(self):
        p=performance(np.array([1,2,3]),np.array([10,20,30]),"regression")
        self.assertLess(p["R2"],0)
        self.assertGreater(p["RMSE"],1)
        self.assertGreater(p["MAPE"],1)

    def test_zero_label_mape_is_missing(self):
        self.assertIsNone(performance(np.array([0,1]),np.array([0,2]),"regression")["MAPE"])

    def test_paper_cr_conflict_remains_visible(self):
        row={"success":True,"selected":"second","optimal":"best","same_scene":True,"quality_gap":0.2,"retries":2,"elapsed_s":1,"MS":0.1}
        stats=summarize([row])
        self.assertAlmostEqual(stats["SR_paper"],80)
        self.assertAlmostEqual(stats["CR_paper_literal"],-20)
        self.assertAlmostEqual(stats["CR_closeness_success"],80)

    def test_failed_trial_does_not_have_zero_ms(self):
        result=summarize([{"success":False,"retries":0,"elapsed_s":1}])
        self.assertIsNone(result["MS"])
        self.assertEqual(result["MS_plot"],1)

    def test_ms_is_euclidean(self):
        self.assertAlmostEqual(similarity({"RMSE":1,"MAPE":2,"R2":3},{"RMSE":0,"MAPE":0,"R2":0}),np.sqrt(14))

    def test_graph_has_only_eligible_cross_partition_edges(self):
        models=skeleton()[:2]
        context={"scene":"D1","features":[0.0]*14,"eligible":[models[0]["id"]]}
        graph=build_graph(models,[context])
        self.assertEqual(graph["adjacency"].tolist(),[[1.0],[0.0]])
        net=BipartiteRegressor(self.config["graph_training"])
        result=net(graph);result.sum().backward()
        self.assertEqual(tuple(result.shape),(2,1))
        self.assertIsNotNone(net.model_project.weight.grad)
        self.assertIsNotNone(net.mlp[0].weight.grad)

    def test_hard_resource_and_software_constraints(self):
        env=self.config["environment"]
        bad={"ram_gb":env["ram_gb"]+1,"os":"different","runtime_version":">=2.0","dependencies":{"absent":">=1"}}
        reasons=mismatch(bad,env)
        for reason in ("ram_gb","os","runtime_version","dependency:absent"):
            self.assertIn(reason,reasons)

    def test_wrong_action_order_is_rejected(self):
        planner=ActionPlanner(None,read_json(Path("configs/rules.json")))
        with self.assertRaises(ValueError):
            planner.validate(["特征提取","数据预处理","回归预测"],"regression")

    def test_signature_integrity_and_non_model_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/"weights.bin";artifact.write_bytes(b"original")
            key=Ed25519PrivateKey.generate();key_file=root/"key.pem"
            key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
            public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
            config=copy.deepcopy(self.config);config["public_keys"]={"lab":base64.b64encode(public).decode()}
            model=skeleton()[0];model.update({"source":"registered-lab","publisher":"lab","artifacts":["weights.bin"]})
            seal(model,root,key_file)
            self.assertEqual(trust_values(model,root,config,[],now=100),(1.0,1.0,1.0))
            history=[{"time":99,"model":model["id"],"cause":"resource","model_fault":None}]
            self.assertEqual(trust_values(model,root,config,history,now=100)[2],1)
            history.append({"time":99,"model":model["id"],"cause":"model","model_fault":"output"})
            self.assertAlmostEqual(trust_values(model,root,config,history,now=100)[2],0.5)
            artifact.write_bytes(b"modified")
            self.assertEqual(trust_values(model,root,config,[],now=100)[1],0)
            model["publisher"]="unknown"
            self.assertLess(trust_values(model,root,config,[],now=100)[0],1)
            with self.assertRaises(ValueError):
                safe_path(root,"../outside.pt")

    def test_group_leakage_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"D1").mkdir()
            values={}
            for split in ("train","val","test"):
                values["X_"+split]=np.ones((2,3))
                values["y_"+split]=np.ones(2)
                values["groups_"+split]=np.array(["same","same"])
            np.savez(root/"D1/data.npz",**values)
            with self.assertRaises(ValueError):
                load_dataset(root,"D1")

    def test_perturbation_keeps_shape_and_scales_amplitude(self):
        x=np.ones((2,32,3),dtype=np.float32)
        self.assertEqual(perturb(x,1.1,1).shape,x.shape)
        np.testing.assert_allclose(perturb(x,1,1.01),x*1.01)


if __name__=="__main__":
    unittest.main()
