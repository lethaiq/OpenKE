import config
import models
import tensorflow as tf
import numpy as np

#++++++++++++++TransE++++++++++++++++++++

con = config.Config()
con.set_in_path("./benchmarks/umls_KE/")
con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_bern(0)
con.set_dimension(100)
con.set_margin(1)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
con.init()
con.set_model(models.TransE)
con.run()
con.test()

