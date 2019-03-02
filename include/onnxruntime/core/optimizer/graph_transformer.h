// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

enum TransformerLevel {
  Default_Global = 0,
  Default_ProviderSpecific = 1,
  Optional_L1 = 2,
  Optional_L2 = 4,  
  MaxTransformerLevel = 8
};

/**
@class GraphTransformer

The interface for in-place transformation of a Graph.
*/
class GraphTransformer {
 public:
  GraphTransformer(const std::string& name, const std::string& desc, const TransformerLevel level, std::vector<std::string>&& providers)
      : name_(name), desc_(desc), level_(level), compatible_providers_{std::move(providers)} {
  }

  virtual ~GraphTransformer() = default;

  /** Gets the name of this graph transformer. */
  const std::string& Name() const noexcept {
    return name_;
  }

  /** Gets the description of this graph transformer. */
  const std::string& Description() const noexcept {
    return desc_;
  }

   /** Gets the level of this graph transformer. */
  const TransformerLevel Level() const noexcept {
    return level_;
  }

  /** Checks whether nodes assigned provider is compatible with this transformer. */
  bool IsProviderCompatible(const std::string& providerType) const {
    if (providerType.empty() || std::find(compatible_providers_.begin(), compatible_providers_.end(), providerType) != compatible_providers_.end()) {
      return true;
    }
    return false;
  }

  /** Apply the in-place transformation defined by this transformer to the provided Graph instance.
  @param[out] modified Set to true if the Graph was modified.
  @returns Status with success or error information.
  */
  common::Status Apply(Graph& graph, bool& modified) const;  

 protected:
  /** Helper method to call ApplyImpl on any subgraphs in the Node. */
  common::Status Recurse(Node& node, bool& modified, int graph_level) const {
    int subgraph_level = ++graph_level;
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& subgraph = *entry.second;
      ORT_RETURN_IF_ERROR(ApplyImpl(subgraph, modified, subgraph_level));
    }

    return Status::OK();
  }
  
 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformer);

  // Apply the transform to the graph.
  // graph_level is 0 for the main graph, and is incremented when descending into the subgraph of a node.
  // You MUST call Recurse for all valid Nodes in the graph to ensure any subgraphs in control flow nodes
  // (Scan/If/Loop) are processed as well.
  // You should avoid calling Graph::Resolve in ApplyImpl unless you are 100% sure it's required. In most cases
  // the call to Graph::Resolve in Apply prior to ApplyImpl being called, and after ApplyImpl fore the main graph
  // completes (if 'modified' is true) should suffice.
  virtual common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level = 0) const = 0;
  
  const std::string name_;
  const std::string desc_;
  const TransformerLevel level_;
  const std::vector<std::string> compatible_providers_;
};

/**
@class RuleBasedGraphTransformer

Rule based graph transformer that provides an API to register rewrite rules, 
and an API to apply all applicable rules to a Graph.

Represents an IGraphTransformer determined by a set of rewrite-rules.
The transformer will apply all the rewrite-rules iteratively as determined by the underlying rewriting-strategy.
Several rewriting-strategies are possible when traversing the graph and applying rewrite rules, 
each with different trade offs. At the moment, we define one that performs top-down traversal of nodes.

@TODO: Is a bottom-up traversal more efficient?
@TODO: Is it worth adding the max number of passes a rule should be applied for?
@TODO: We need to define a contract about whether a rewrite rule is allowed to leave
       the graph in an inconsistent state (this will determine when and where we will be
       calling Graph::resolve().
*/
class RuleBasedGraphTransformer : public GraphTransformer {
 public:
  RuleBasedGraphTransformer(const std::string& name, const std::string& desc, const TransformerLevel level, std::vector<std::string>&& providers)
      : GraphTransformer(name, desc, level, std::move(providers)){}

  /**
  Register a rewriting rule.

  @TODO (revisit needed): Using OpSignature* here will ask that OpSignature should be stored globally. 
  Otherwise, there will be multiple addresses/pointers for the same operator or function. 
  To avoid this, we may use OpSignature ID as the key, which should be name_domain_version.
  We will use the string type instead of the OpSchema for now. We should probably add a version as well.
  */
  Status Register(const std::string& op_type, std::unique_ptr<RewriteRule> rule);

  /** Check if the given op_type has any rules registered for it 
  @returns true if there are rules registered for this op_type.*/
  bool HasRules(const std::string& op_type) const {
    return op_to_rules_.find(op_type) != op_to_rules_.cend();
  }

  /**
  Gets the rewrite rules for the given op_type.
  @returns a pointer to the vector containing all the rewrite rules registered for op_type if found. nullptr
  otherwise.
  */
  const std::vector<std::unique_ptr<RewriteRule>>* GetRewriteRules(const std::string& op_type) const {
    auto entry = op_to_rules_.find(op_type);
    if (entry != op_to_rules_.cend())
      return &entry->second;

    return nullptr;
  }

 private:
  using RewriteRuleSet = std::unordered_map<std::string, std::vector<std::unique_ptr<RewriteRule>>>;

  RewriteRuleSet op_to_rules_;
};

/**
@class TopDownRuleBasedTransformer

This is a rule-based Graph transformer that applies rules by performing top-down passes of the Graph.
*/
class TopDownRuleBasedTransformer : public RuleBasedGraphTransformer {
 public:
  TopDownRuleBasedTransformer(const std::string& name, const std::string& desc, const TransformerLevel level, std::vector<std::string> providers)
      : RuleBasedGraphTransformer(name, desc, level, std::move(providers)) {}

 private:
  // Performs a single top-down traversal of the graph and applies all registered rules.
  common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
