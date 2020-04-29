# TF Federated Aggregators Placement

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [TF Federated Aggregators Placement](https://github.com/tensorflow/community/pull/NNN) (TODO update when you have community PR #)|
| **Author(s)** | Jason Mancuso (jason@capeprivacy.com)                |
| **Sponsor**   | Michael Reneer (michaelreneer@google.com)            |
| **Updated**   | YYYY-MM-DD  (TODO)                                   |

## Objective

This document proposes adding an explicit `tff.AGGREGATORS` placement to the
Federated Core (FC) in TensorFlow Federated (TFF). This would allow users to express
custom aggregation protocols directly in TFF without adding unnecessary
assumptions about where aggregation functions must be placed and computed.

## Motivation

*Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?*

When approaching federated learning with an eye for security or privacy, it is
useful to divide federated computation into two categories: computations performing
aggregations, and computations performing on-device computation.  Security and
privacy issues tend to show up during the aggregation phase. This is particularly
clear when looking at common methods of adding security guarantees to traditional,
parameter-server style federated learning, for example with secure aggregation or
differentially private federated averaging (DP-FedAvg).

In security-heightened settings, it is often worthwhile to separate computation
done in this aggregation phase from computation performed by the server in the
traditional parameter server setup. This amounts to delegating aggregations to a
third-party service. For example, when the clients are mistrustful of the server,
aggregations might be delegated to a trusted execution environment or to  a cluster
of machines engaging in a secure multi-party computation protocol. Another example
is secure aggregation in the
[Encode-Shuffle-Analyze (ESA)](https://arxiv.org/abs/1710.00901)
model, which in a federated context generally assumes an additional
party to perform the secure shuffling needed to realize a differential privacy
guarantee. Since this is an established area of the literature with strong
motivations and results, we see this as an important line of work for TFF to
support in order to keep with its
[project goals](https://github.com/tensorflow/federated#tensorflow-federated).

In general, any secure aggregation protocol can be represented as a coordinated
computation between three groups of parties: a server, a (potentially singleton)
set of aggregators, and a set of clients. Note that these need not be mutually
exclusive, so for example the traditional parameter server setting can be recovered
as a special case by treating the server as a singleton aggregators set.

The TFF Federated Core (FC) language currently realizes logically-distinct parties
as "placements". While there exist `tff.SERVER` and `tff.CLIENTS` placements in FC,
there is no `tff.AGGREGATORS` placement. Without such a placement, implementing new
aggregation protocols in TFF can require low-level programming of the TFF executor
stacks, as evidenced by
[this community attempt to integrate secure aggregation](https://github.com/tf-encrypted/rfcs/blob/master/20190924-tensorflow-federated/integration-strategies.md).
By adding a `tff.AGGREGATORS` placement, users can more easily implement new
aggregation protocols by expressing them as federated computations in FC.

## User Benefit

*How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post?*

Users can now express custom aggregation protocols in the Federated Core by working
with federated data placed on `tff.AGGREGATORS`. Users will be unencumbered by the
constraints of the current federated types in FC.

## Design Proposal

*This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Make sure you’ve thought through and addressed the following sections. If a section is not relevant to your specific proposal, please explain why, e.g. your RFC addresses a convention or process, not an API.*


### Alternatives Considered
A lower effort alternative might be to expect users to write custom executors, or custom executor stacks, to include additional "aggregator" parties when executing intrinisics. AGGREGATORS would stay outside of the FC type system, but could be sill be included in federated computations. This allows library designers to extend TFF for their own use cases. This is a major disadvantage, since users are only expected to be familiar with TFF Federated Learning (FL) or FC APIs, and this is a feature that would be useful to the majority of TFF users.

### Performance Implications
* Do you expect any (speed / memory)? How will you confirm?
* There should be microbenchmarks. Are there?
* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

### Dependencies
* Dependencies: does this proposal add any new dependencies to TensorFlow?
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

### Engineering Impact
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

### Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged. 

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
