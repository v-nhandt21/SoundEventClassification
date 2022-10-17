class Model(nn.Module):
     def __init__(self, n_student_blocks, n_emotion_classes):
          super(Model, self).__init__()
          self.backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', gradient_checkpointing = False)
          self.backbone.config.mask_time_prob = 0.3
          self.backbone.config.mask_time_min_masks = 5
          self.backbone.config.mask_feature_prob = 0.3
          self.backbone.config.mask_feature_min_masks = 2


          self.backbone.feature_extractor._freeze_parameters()
          self.n_student_blocks = n_student_blocks
          # Size of input and output features of the bottlenecks:  
          # [B, n_features_in] -> [B, n_x_bottleneck_out]
          self.n_features_in = 768
          self.n_gender_bottleneck_out = 128
          self.n_student_bottleneck_out = 256 
          self.n_teacher_bottleneck_out = 128
          self.n_gender_classes = 1
          self.n_emotion_classes = n_emotion_classes

          self.gender_block = Block(  n_classes = self.n_gender_classes, 
                                        n_features_in = self.n_features_in, 
                                        n_bottleneck_out = self.n_gender_bottleneck_out, 
                                        fusion_block = False  )
          
          self.student_blocks = nn.ModuleList([
               Block(  n_classes = self.n_emotion_classes, 
                         n_features_in = self.n_features_in, 
                         n_bottleneck_out = self.n_student_bottleneck_out, 
                         fusion_block = False    ) 
                         for i in range(self.n_student_blocks)
          ])

          self.teacher_block =  Block(    n_classes = self.n_emotion_classes, 
                                             n_features_in = self.n_features_in, 
                                             n_bottleneck_out = self.n_teacher_bottleneck_out, 
                                             fusion_block = False    ) 

          self.fusion_block =  Block(    n_classes = self.n_emotion_classes, 
                                             n_features_in = self.n_features_in * (self.n_student_blocks+1), 
                                             n_bottleneck_out = None, 
                                             fusion_block = True    ) 
     
     def forward(self, x, alpha = 0, is_gender_only = False, use_distill = False):
          x = self.backbone(x, output_hidden_states = True, return_dict = True)
          # Process the gender branch, if is_gender_only is True, we dont need further process
          # This will speed up training procedure
          # Apply Reverselayer to the last_hidden_state
          if is_gender_only:
               reverse_feature = ReverseLayerF.apply(torch.mean(x.last_hidden_state, dim=1), alpha)
               gender_logit = self.gender_block(reverse_feature)[1]
               return gender_logit

          hidden_features = []
          output_features = ()
          emotion_logits = ()

          # Process the Student blocks
          for i in range(self.n_student_blocks):
               hidden_feature = torch.mean(x.hidden_states[i-self.n_student_blocks-1], dim=1)
               # Keep the input feature that is fed to student block
               hidden_features = hidden_features + [hidden_feature] 
               # Also keep the outputs of student blocks
               st_output_feature, st_emotion_logit = self.student_blocks[i](hidden_feature)
               output_features = output_features + (st_output_feature, )
               emotion_logits = emotion_logits + (st_emotion_logit, )

          # Process the Teacher block
          last_hidden_feature = torch.mean(x.last_hidden_state, dim=1)
          hidden_features = hidden_features + [last_hidden_feature]
          tc_output_features, tc_emotion_logit = self.teacher_block(last_hidden_feature)
          output_features = output_features + (tc_output_features, )
          emotion_logits = emotion_logits + (tc_emotion_logit, )

          # Process the Fusion block
          fusion_feature = torch.cat(hidden_features, dim=-1) # Concatenate input features of student and teacher blocks
          fusion_logit = self.fusion_block(fusion_feature)[1]
          output_features = output_features + (None, ) # We dont process the output feature of fusion head, so leave it None
          emotion_logits = emotion_logits + (fusion_logit, )

          if not use_distill:
               return [tc_emotion_logit], None
          else:
               return list(emotion_logits), list(output_features)