"""
Adaptive Fusion Mechanism for Two-Stream Emotion Recognition
Giải pháp cho Spatial Attention Misalignment và Semantic Asymmetry

Thành phần:
1. SpatialAlignmentAnalyzer: Phát hiện misalignment giữa attention maps
2. ModalityUncertaintyEstimator: Ước lượng độ tin cậy (uncertainty) của từng modality
3. ConflictDetector: Phát hiện xung đột semantic giữa hai nhánh
4. AdaptiveModalityGating: Gating động dựa trên uncertainty
5. SemanticBridgingModule: Cầu nối bất đối xứng ngữ nghĩa
6. AdaptiveFusionModule: Module fusion chính
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialAlignmentAnalyzer(nn.Module):
    """
    Phân tích mức độ giao nhau (alignment) của spatial attention maps giữa hai modality.
    
    Đầu ra:
    - alignment_score: [0, 1] - 1 = hoàn toàn aligned, 0 = hoàn toàn misaligned
    - conflict_mask: Bản đồ các vùng xung đột
    - alignment_map: Heatmap alignment chi tiết
    """
    
    def __init__(self, embed_dim=512, num_patches=49):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Attention map generator - chiếu features thành spatial attention
        self.attention_generator_ir = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_patches)
        )
        
        self.attention_generator_lm = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_patches)
        )
        
    def compute_alignment_score(self, attn_ir, attn_lm):
        """
        Tính alignment score dựa trên cosine similarity và overlap của attention peaks.
        
        Args:
            attn_ir: [B, N] - normalized attention map từ IR50
            attn_lm: [B, N] - normalized attention map từ Landmark
            
        Returns:
            alignment_score: [B] - scalar score [0, 1]
            overlap_ratio: [B] - tỉ lệ patches có top-k attention trùng nhau
        """
        B = attn_ir.shape[0]
        
        # Cosine similarity giữa hai attention maps
        cosine_sim = F.cosine_similarity(attn_ir.unsqueeze(1), attn_lm.unsqueeze(1), dim=-1)
        cosine_sim = (cosine_sim + 1) / 2  # Normalize to [0, 1]
        
        # Kiểm tra overlap của top-k patches
        k = max(1, self.num_patches // 4)  # Top 25% patches
        
        _, top_ir = torch.topk(attn_ir, k, dim=1)  # [B, k]
        _, top_lm = torch.topk(attn_lm, k, dim=1)  # [B, k]
        
        # Vectorized computation using masks
        # Create one-hot masks for top-k positions
        mask_ir = torch.zeros_like(attn_ir)  # [B, N]
        mask_lm = torch.zeros_like(attn_lm)  # [B, N]
        
        mask_ir.scatter_(1, top_ir, 1.0)  # Mark top-k positions
        mask_lm.scatter_(1, top_lm, 1.0)
        
        # Compute intersection (element-wise multiplication)
        overlap_mask = mask_ir * mask_lm  # [B, N]
        overlap = overlap_mask.sum(dim=1) / k  # [B]
        
        # Combine cosine similarity và overlap
        alignment_score = 0.6 * cosine_sim.mean(dim=-1) + 0.4 * overlap
        
        return alignment_score, overlap
    
    def forward(self, feat_ir, feat_lm):
        """
        Args:
            feat_ir: [B, N, C] - features từ IR50
            feat_lm: [B, N, C] - features từ Landmark
            
        Returns:
            alignment_score: [B] - overall alignment
            attn_ir: [B, N] - attention map IR50
            attn_lm: [B, N] - attention map Landmark
            conflict_mask: [B, N] - confidence mask của vùng xung đột
        """
        B = feat_ir.shape[0]
        
        # Generate attention maps
        attn_ir = self.attention_generator_ir(feat_ir)  # [B, N]
        attn_lm = self.attention_generator_lm(feat_lm)  # [B, N]
        
        # Normalize attention maps
        attn_ir_norm = F.softmax(attn_ir, dim=-1)
        attn_lm_norm = F.softmax(attn_lm, dim=-1)

        if attn_ir_norm.dim() == 3:
            attn_ir_norm = attn_ir_norm.mean(dim=1)
        if attn_lm_norm.dim() == 3:
            attn_lm_norm = attn_lm_norm.mean(dim=1)
        
        # Compute alignment score
        alignment_score, overlap = self.compute_alignment_score(attn_ir_norm, attn_lm_norm)
        
        # Compute conflict mask (element-wise difference)
        conflict_mask = torch.abs(attn_ir_norm - attn_lm_norm)  # [B, N]
        
        return {
            'alignment_score': alignment_score,  # [B]
            'attn_ir': attn_ir_norm,  # [B, N]
            'attn_lm': attn_lm_norm,  # [B, N]
            'conflict_mask': conflict_mask,  # [B, N]
            'overlap_ratio': overlap  # [B]
        }


class ModalityUncertaintyEstimator(nn.Module):
    """
    Ước lượng độ tin cậy (uncertainty/confidence) của từng modality.
    
    Dựa trên:
    1. Entropy của attention distributions
    2. Feature activation strength
    3. Modality-specific discriminability
    """
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # IR50 uncertainty head (appearance features)
        self.ir_uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Landmark uncertainty head (geometric features)
        self.lm_uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_entropy(self, probs):
        """
        Tính entropy của probability distribution.
        Entropy cao = uncertainty cao, low confidence
        
        Args:
            probs: [B, N] - normalized probabilities
            
        Returns:
            entropy: [B] - entropy values [0, 1]
        """
        if probs.dim() > 2:
            probs = probs.flatten(start_dim=1)

        # Clamp để tránh log(0)
        probs = torch.clamp(probs, 1e-7, 1.0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # [B]
        
        # Normalize entropy to [0, 1]
        # Max entropy for uniform distribution = log(N)
        max_entropy = np.log(probs.shape[-1])
        entropy_norm = entropy / max_entropy
        
        return entropy_norm
    
    def compute_feature_strength(self, features):
        """
        Tính độ mạnh (activation strength) của features.
        
        Args:
            features: [B, N, C] - feature maps
            
        Returns:
            strength: [B] - activation strength [0, 1]
        """
        if features.dim() > 3:
            features = features.flatten(start_dim=1)

        # L2 norm across feature dimension
        feature_norm = torch.norm(features, p=2, dim=-1)  # [B, N]
        
        # Average pooling
        strength = torch.mean(feature_norm, dim=-1)  # [B]
        
        # Normalize to [0, 1] using sigmoid
        strength = torch.sigmoid(strength)
        
        return strength
    
    def forward(self, feat_ir, feat_lm, attn_ir, attn_lm):
        """
        Args:
            feat_ir: [B, N, C] - features từ IR50
            feat_lm: [B, N, C] - features từ Landmark
            attn_ir: [B, N] - attention map IR50
            attn_lm: [B, N] - attention map Landmark
            
        Returns:
            uncertainty_ir: [B] - uncertainty của IR50 (1 = very uncertain)
            uncertainty_lm: [B] - uncertainty của Landmark (1 = very uncertain)
            confidence_ir: [B] - confidence = 1 - uncertainty_ir
            confidence_lm: [B] - confidence = 1 - uncertainty_lm
        """
        B = feat_ir.shape[0]
        
        # Extract class tokens
        cls_ir = feat_ir[:, 0, :]  # [B, C]
        cls_lm = feat_lm[:, 0, :]  # [B, C]
        
        # Entropy-based uncertainty
        entropy_ir = self.compute_entropy(attn_ir)  # [B]
        entropy_lm = self.compute_entropy(attn_lm)  # [B]
        
        # Feature strength
        strength_ir = self.compute_feature_strength(feat_ir)  # [B]
        strength_lm = self.compute_feature_strength(feat_lm)  # [B]
        
        # Learned uncertainty estimation
        learned_unc_ir = self.ir_uncertainty_head(cls_ir).squeeze(-1)  # [B]
        learned_unc_lm = self.lm_uncertainty_head(cls_lm).squeeze(-1)  # [B]
        
        # Combine: high entropy + low strength = high uncertainty
        entropy_ir = entropy_ir.reshape(B)
        entropy_lm = entropy_lm.reshape(B)
        strength_ir = strength_ir.reshape(B)
        strength_lm = strength_lm.reshape(B)
        learned_unc_ir = learned_unc_ir.reshape(B)
        learned_unc_lm = learned_unc_lm.reshape(B)

        uncertainty_ir = 0.4 * entropy_ir + 0.3 * (1 - strength_ir) + 0.3 * learned_unc_ir
        uncertainty_lm = 0.4 * entropy_lm + 0.3 * (1 - strength_lm) + 0.3 * learned_unc_lm
        
        confidence_ir = 1.0 - uncertainty_ir
        confidence_lm = 1.0 - uncertainty_lm
        
        return {
            'uncertainty_ir': uncertainty_ir,  # [B]
            'uncertainty_lm': uncertainty_lm,  # [B]
            'confidence_ir': confidence_ir,   # [B]
            'confidence_lm': confidence_lm,   # [B]
            'entropy_ir': entropy_ir,
            'entropy_lm': entropy_lm,
            'strength_ir': strength_ir,
            'strength_lm': strength_lm
        }


class ConflictDetector(nn.Module):
    """
    Phát hiện xung đột (conflict) giữa hai modality.
    
    Xung đột xảy ra khi:
    1. Hai modality hướng tới các quyết định phân loại trái ngược
    2. Attention maps không giao nhau
    3. Feature representations phân kỳ mạnh
    """
    
    def __init__(self, embed_dim=512, num_classes=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Lightweight classifiers cho từng modality
        self.classifier_ir = nn.Linear(embed_dim, num_classes)
        self.classifier_lm = nn.Linear(embed_dim, num_classes)
        
    def compute_prediction_conflict(self, logits_ir, logits_lm):
        """
        Tính mức độ xung đột dự đoán giữa hai modality.
        
        Args:
            logits_ir: [B, num_classes]
            logits_lm: [B, num_classes]
            
        Returns:
            conflict: [B] - conflict score [0, 1]
        """
        # Softmax để lấy probabilities
        probs_ir = F.softmax(logits_ir, dim=-1)
        probs_lm = F.softmax(logits_lm, dim=-1)
        
        # Lấy top-1 predictions
        pred_ir = torch.argmax(probs_ir, dim=-1)
        pred_lm = torch.argmax(probs_lm, dim=-1)
        
        # Mismatch: prediction khác nhau
        mismatch = (pred_ir != pred_lm).float()  # [B]
        
        # Confidence difference
        conf_ir = torch.max(probs_ir, dim=-1)[0]
        conf_lm = torch.max(probs_lm, dim=-1)[0]
        conf_diff = torch.abs(conf_ir - conf_lm)  # [B]
        
        # KL divergence (một measure của sự khác nhau)
        kl_ir_to_lm = F.kl_div(torch.log(probs_lm + 1e-7), probs_ir, reduction='none').sum(dim=-1)
        kl_lm_to_ir = F.kl_div(torch.log(probs_ir + 1e-7), probs_lm, reduction='none').sum(dim=-1)
        kl_div = (kl_ir_to_lm + kl_lm_to_ir) / 2
        kl_div_norm = torch.sigmoid(kl_div)  # Normalize to [0, 1]
        
        # Combine conflict signals
        conflict = 0.3 * mismatch + 0.3 * conf_diff + 0.4 * kl_div_norm
        
        return conflict, {
            'mismatch': mismatch,
            'conf_diff': conf_diff,
            'kl_div': kl_div_norm,
            'pred_ir': pred_ir,
            'pred_lm': pred_lm
        }
    
    def forward(self, feat_ir, feat_lm, attn_ir, attn_lm):
        """
        Args:
            feat_ir: [B, N, C]
            feat_lm: [B, N, C]
            attn_ir: [B, N]
            attn_lm: [B, N]
            
        Returns:
            conflict_score: [B]
            details: dict chứa chi tiết phát hiện xung đột
        """
        # Class token
        cls_ir = feat_ir[:, 0, :]  # [B, C]
        cls_lm = feat_lm[:, 0, :]  # [B, C]
        
        # Intermediate predictions
        logits_ir = self.classifier_ir(cls_ir)  # [B, num_classes]
        logits_lm = self.classifier_lm(cls_lm)  # [B, num_classes]
        
        # Prediction conflict
        pred_conflict, pred_details = self.compute_prediction_conflict(logits_ir, logits_lm)
        pred_conflict = pred_conflict.reshape(-1)
        
        # Attention map conflict (từ alignment analyzer)
        attn_diff = torch.abs(attn_ir - attn_lm)
        attn_conflict = attn_diff.flatten(start_dim=1).mean(dim=1)  # [B]
        
        # Feature divergence (cosine similarity)
        feat_sim = F.cosine_similarity(cls_ir, cls_lm, dim=-1)  # [B] - [-1, 1]
        feat_conflict = 1.0 - (feat_sim + 1) / 2  # [0, 1]
        feat_conflict = feat_conflict.reshape(-1)
        
        # Combine
        total_conflict = 0.4 * pred_conflict + 0.3 * attn_conflict + 0.3 * feat_conflict
        
        return {
            'conflict_score': total_conflict,  # [B]
            'pred_conflict': pred_conflict,
            'attn_conflict': attn_conflict,
            'feat_conflict': feat_conflict,
            'pred_details': pred_details,
            'logits_ir': logits_ir,
            'logits_lm': logits_lm
        }


class SemanticBridgingModule(nn.Module):
    """
    Cầu nối bất đối xứng ngữ nghĩa giữa Appearance Features (IR50) và Geometric Features (Landmark).
    
    Chiến lược:
    1. Semantic Projection: Chiếu geometric features vào semantic space của appearance
    2. Context Enrichment: Bổ sung context ngữ nghĩa cho geometric features
    3. Cross-modal Anchoring: Sử dụng appearance features làm anchor để hướng geometric
    """
    
    def __init__(self, embed_dim=512, num_patches=49):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Semantic projection network
        self.semantic_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Context enrichment for landmarks
        self.context_enricher = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Cross-modal guidance
        self.cross_modal_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat_ir, feat_lm, alignment_score):
        """
        Args:
            feat_ir: [B, N, C] - Appearance features (high-level semantics)
            feat_lm: [B, N, C] - Geometric features (low-level)
            alignment_score: [B] - Overall alignment score
            
        Returns:
            feat_lm_bridged: [B, N, C] - Enriched geometric features
            bridge_weight: [B] - Bridging strength [0, 1]
        """
        B, N, C = feat_lm.shape
        
        # Semantic projection của landmark features
        feat_lm_proj = self.semantic_projector(feat_lm)  # [B, N, C]
        
        # Context enrichment: use appearance features as context
        feat_lm_enhanced, _ = self.context_enricher(
            feat_lm_proj,
            feat_ir,
            feat_ir,
            need_weights=False
        )

        # Token-wise cross-modal guidance
        bridge_gate = self.cross_modal_gate(
            torch.cat([feat_lm_proj, feat_ir], dim=-1)
        )  # [B, N, 1]
        bridge_gate = bridge_gate.mean(dim=1, keepdim=True)  # [B, 1, 1]
        
        # Bình thường hóa alignment_score thành [0, 1]
        alignment_score_norm = alignment_score.clamp(0, 1)  # [B]
        
        # Dynamic bridging weight
        # - Khi aligned tốt: giữ nguyên landmark features (bridge weight thấp)
        # - Khi misaligned: dùng semantic features để hướng dẫn (bridge weight cao)
        bridge_weight = (1.0 - alignment_score_norm).view(B, 1, 1)  # [B, 1, 1]
        
        # Interpolation
        feat_lm_bridged = feat_lm + bridge_weight * bridge_gate * (feat_lm_enhanced - feat_lm)
        
        return {
            'feat_lm_bridged': feat_lm_bridged,
            'bridge_weight': bridge_weight.squeeze(-1).squeeze(-1),
            'feat_lm_proj': feat_lm_proj
        }


class AdaptiveModalityGating(nn.Module):
    """
    Cơ chế gating động để phân bổ trọng số của hai modality dựa trên:
    1. Individual confidence/uncertainty của từng modality
    2. Alignment/conflict status giữa hai modality
    3. Classification uncertainty (entropy)
    
    Output: Dynamic weights [w_ir, w_lm] sao cho w_ir + w_lm = 1
    """
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Gating network
        self.gating_net = nn.Sequential(
            nn.Linear(4, 32),  # Input: [conf_ir, conf_lm, alignment_score, conflict_score]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Output: logits for [w_ir, w_lm]
        )
        
    def forward(self, confidence_ir, confidence_lm, alignment_score, conflict_score):
        """
        Args:
            confidence_ir: [B] - Confidence của IR50
            confidence_lm: [B] - Confidence của Landmark
            alignment_score: [B] - Alignment score [0, 1]
            conflict_score: [B] - Conflict score [0, 1]
            
        Returns:
            weight_ir: [B] - Trọng số cho IR50 [0, 1]
            weight_lm: [B] - Trọng số cho Landmark [0, 1]
            gating_details: dict chứa chi tiết gating
        """
        B = confidence_ir.shape[0]
        
        # Stack inputs
        gating_input = torch.stack([
            confidence_ir,
            confidence_lm,
            alignment_score,
            conflict_score
        ], dim=-1)  # [B, 4]
        
        # Gating logits
        gating_logits = self.gating_net(gating_input)  # [B, 2]
        
        # Softmax để lấy weights
        weights = F.softmax(gating_logits, dim=-1)  # [B, 2]
        weight_ir = weights[:, 0]  # [B]
        weight_lm = weights[:, 1]  # [B]
        
        # Decision reasoning
        # Nếu IR50 confident: tăng w_ir
        # Nếu Landmark confident: tăng w_lm
        # Nếu conflict cao: giảm weights cả hai (uncertainty)
        
        return {
            'weight_ir': weight_ir,
            'weight_lm': weight_lm,
            'gating_logits': gating_logits,
            'gating_input': gating_input
        }


class AdaptiveFusionModule(nn.Module):
    """
    Module fusion chính kết hợp tất cả các thành phần.
    Giải quyết:
    1. Spatial attention misalignment
    2. Semantic asymmetry
    3. Dynamic modality weighting
    4. Conflict resolution
    """
    
    def __init__(self, embed_dim=512, num_patches=49, num_classes=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_classes = num_classes
        
        # Components
        self.alignment_analyzer = SpatialAlignmentAnalyzer(embed_dim, num_patches)
        self.uncertainty_estimator = ModalityUncertaintyEstimator(embed_dim)
        self.conflict_detector = ConflictDetector(embed_dim, num_classes)
        self.semantic_bridger = SemanticBridgingModule(embed_dim, num_patches)
        self.gating = AdaptiveModalityGating(embed_dim)
        
        # Final fusion & classification
        self.fusion_projector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, feat_ir, feat_lm, return_intermediate=False):
        """
        Args:
            feat_ir: [B, N, C] - Features từ IR50
            feat_lm: [B, N, C] - Features từ Landmark
            return_intermediate: bool - Có return intermediate results không
            
        Returns:
            fused_feat: [B, C] - Fused feature representation
            intermediate_results: dict (nếu return_intermediate=True)
        """
        B = feat_ir.shape[0]
        
        # 1. Spatial Alignment Analysis
        alignment_result = self.alignment_analyzer(feat_ir, feat_lm)
        alignment_score = alignment_result['alignment_score']
        attn_ir = alignment_result['attn_ir']
        attn_lm = alignment_result['attn_lm']
        
        # 2. Uncertainty Estimation
        uncertainty_result = self.uncertainty_estimator(feat_ir, feat_lm, attn_ir, attn_lm)
        confidence_ir = uncertainty_result['confidence_ir']
        confidence_lm = uncertainty_result['confidence_lm']
        
        # 3. Conflict Detection
        conflict_result = self.conflict_detector(feat_ir, feat_lm, attn_ir, attn_lm)
        conflict_score = conflict_result['conflict_score']
        
        # 4. Semantic Bridging
        bridging_result = self.semantic_bridger(feat_ir, feat_lm, alignment_score)
        feat_lm_bridged = bridging_result['feat_lm_bridged']
        
        # 5. Adaptive Gating
        gating_result = self.gating(confidence_ir, confidence_lm, alignment_score, conflict_score)
        weight_ir = gating_result['weight_ir']  # [B]
        weight_lm = gating_result['weight_lm']  # [B]
        
        # 6. Weighted Fusion
        cls_ir = feat_ir[:, 0, :]  # [B, C]
        cls_lm_bridged = feat_lm_bridged[:, 0, :]  # [B, C]
        
        weighted_ir = cls_ir * weight_ir.unsqueeze(1)  # [B, C]
        weighted_lm = cls_lm_bridged * weight_lm.unsqueeze(1)  # [B, C]
        
        fused = torch.cat([weighted_ir, weighted_lm], dim=-1)  # [B, C*2]
        fused_feat = self.fusion_projector(fused)  # [B, C]
        
        if return_intermediate:
            return fused_feat, {
                'alignment': alignment_result,
                'uncertainty': uncertainty_result,
                'conflict': conflict_result,
                'gating': gating_result,
                'bridging': bridging_result,
                'weight_ir': weight_ir,
                'weight_lm': weight_lm,
                'alignment_score': alignment_score,
                'conflict_score': conflict_score
            }
        
        return fused_feat
