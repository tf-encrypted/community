# TF Federated Aggregators Placement

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | https://github.com/tensorflow/community/pull/TODO    |
| **Author(s)** | Jason Mancuso (jason@capeprivacy.com)                |
| **Sponsor**   | Michael Reneer (michaelreneer@google.com)            |
| **Updated**   | 2020-05-04                                           |

## Objective

This document proposes adding a `tff.AGGREGATORS` placement to the Federated Core
(FC) in TensorFlow Federated (TFF). This would lift the requirement that all
aggregations be computed on `tff.SERVER` while still allowing users to express
custom aggregation logic using FC & TF.

## Motivation

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
By adding a new `tff.AGGREGATORS` placement, users can more easily implement new
aggregation protocols by expressing them as federated computations in FC.

## User Benefit

Users can now express custom aggregation protocols in the Federated Core by working
with federated data placed on `tff.AGGREGATORS`. Users will be unencumbered by the
constraints of the current federated types in FC.

## Design Proposal

Adding the `tff.AGGREGATORS` placement for federated types involves adding a new
`Placement` and `PlacementLiteral`, and then extending the compiler to recognize
federated values with this placement when computing intrinsics. The compiler
generally defines separate intrinsics by placement; e.g. 
`tff.federated_value(value, placement)` is actually interpreted by the compiler as
`federated_value_at_clients(value)` or `federated_value_at_server(value)`,
depending on the provided `placement`. This means we we will want to add new
intrinsics that correspond to `tff.AGGREGATORS`, e.g.
`federated_value_at_aggregators`.

Existing federated computation that will need modification fall into the two
categories below:

1. Intrinsics for federated computations that are already parameterized by
placement. Note some of these functions don't have a `placement` arg in their
public API signature, but internally correspond to different IntrinsicDefs based on
placement of their federated input(s).
    - `federated_eval`
    - `federated_map`
    - `federated_value`
    - `federated_zip`
    - `sequence_map`
2. Intrinsics that will need to be parameterized by placement, but currently
aren't. 
    - `federated_aggregate`
    - `federated_broadcast`
    - `federated_collect`
    - `federated_mean`
    - `federated_reduce`
    - `federated_secure_sum`
    - `federated_sum`
    - `sequence_reduce`
    - `sequence_sum`

Intrinsics in the latter category will likely need further discussion. This is
because implementation details could change aspects of the underlying "federated
algebra", like closure, or could introduce subtle semantic changes.

As an example, assume we extend `federated_collect` to handle signatures of
`T@CLIENTS -> T*@AGGREGATORS` and `T@AGGREGATORS -> T*@SERVER` (in addition to the
current `CLIENTS -> SERVER`). If we want to maintain algebraic closure, we would
extend `federated_broadcast` to handle `T@SERVER -> {T}@AGGREGATORS` and
`T@AGGREGATORS -> {T}@CLIENTS`; similarly, we would extend `sequence_reduce` to
handle values of type `T@AGGREGATORS`. In this scenario, the new
`federated_broadcast` would be a natural generalization of the old, however it's
not clear if this kind of semantic change would be confusing to users of the FC.

We hope this will be a good starting point for discussion. Ultimately, the RFC
process should allow us to elaborate the exact type signatures that each of the new
IntrinsicDefs should satisfy.

### Alternatives Considered
A lower effort alternative might be to expect users to write custom executors, or
custom executor stacks, to include additional "aggregator" parties when executing
intrinisics. "AGGREGATORS" would stay outside of the FC type system, but could
still be included in federated computations. This might allow library designers to
extend TFF for their own use cases, but hinders the majority of TFF users who are
not expected to learn the executor API.

We also briefly considered the name `tff.AGGREGATOR` instead of `tff.AGGREGATORS`.
We decided on the latter for two reasons:
1. `tff.AGGREGATOR` does not capture the possibility of multiple executor stacks
coordinating aggregation (the existing `ComposingExecutor` qualifies as one such
case).
2. `tff.AGGREGATOR` is equivalent to a singleton `tff.AGGREGATORS`.

### Performance Implications
This is an additive improvement to the FC, so there should be no performance
implications for existing functionality. TFF is designed to support this kind of
addition with minimal overhead. New functionality could be less performant relative
to current practices, but only from overhead inherent to adding a new node to a
distributed computation.

### Dependencies
This change brings no new dependencies. Since this proposal adds a new federated
type, any project that enforces limits based on the current federated types may
have to be updated. We will work with the TFF team to identify any affected
projects and limit any breaking changes.

### Engineering Impact
This code will likely bring marginal increases to build and test time, but changes
to binary size should be negligible. Executor factories including a stack for the
`tff.AGGREGATORS` placement will experience a nontrivial increase in startup time,
but not all executor factories will need to include a stack for this placement.

The code for this change will be mixed into existing modules in the TF Federated
core. Since it affects the type system used by the TFF compiler and requires that
relevant intrinsic definitions be modified to recognize a new placement, it will
touch many different places in the TFF stack. Those who already own and maintain
those code units will maintain and improve the change in the future, which makes
their feedback critical throughout design and implementation.

### Best Practices
The new `Placement` for federated types brings an addition to the Federated Core,
which will be communicated in the TFF API documentation. This will only be relevant
for users of the lower-level Federated Core, at least until a higher level API is
included that relies on it. Below, we detail how this change should be communicated
by existing tutorials.

### Tutorials and Examples
Since this is a modification of an existing API, it likely does not warrant a new
tutorial. We instead suggest modifying the existing
[Part 1 Federated Core tutorial (FC 1)](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1)
to include one or more federated computations that operate on Aggregator-placed
data. We also considered modifying part 2 of the FC tutorial, but decided against
that due to its stated goals.

Concretely, we recommend two modifications to the FC 1 tutorial:
- In the "Placement" section, the discussion will need to include the
`tff.AGGREGATORS` placement. This section should stress that the placement can be
considered optional, whereas the others (`tff.CLIENTS`, `tff.SERVER`) are strictly
necessary for most interesting federated computations.
- In the "Composing Federated Computations" section, we recommend adding a short
sub-section or paragraph that describes how one might refactor the
`get_average_temperature` function to perform its `federated_mean` with a placement
of `tff.AGGREGATORS`. We include short and long form examples below for
consideration.

```python
# short form
@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
    averaged_temp = tff.federated_mean(sensor_readings, placement=tff.AGGREGATORS)
    return tff.federated_collect(averaged_temp, placement=tff.SERVER)

# long form
@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
    collected_readings = tff.federated_collect(sensor_readings, placement=tff.AGGREGATORS)
    num_clients = len(collected_readings)
    total_temp = tff.sequence_sum(collected_readings)
    return tff.federated_map(lambda x: x / num_clients, total_temp)
```

### Compatibility

Since this design adds new functionality, it would change the public API. While TFF
is still pre-1.0, it does not yet explicitly guarantee backwards compatibility of
its public API. Nevertheless, we can hope to limit impact on the public API through
judicious use of default keyword arguments.

Concretely, we can maintain backwards compatibility for federated computations that
gain a `placement` keyword argument by defaulting that argument to `tff.SERVER`. We
recommend _judicious_ use because there may be instances where a change in semantic
justifies a breaking change. These should be taken on a case-by-case basis, and we
hope to clearly define and justify any breaking changes that might arise.

This design does not significantly impact compatibility with the rest of the TF
ecosystem.

## Questions and Discussion Topics

- Which of the intrinsics above should actually be modified/parameterized?
- How strict should we be about algebraic closure in the federated type system?
There could be an argument against, e.g. if we want to limit which intrinsics can
ever involve `tff.AGGREGATORS`.
- Are the existing tutorial changes sufficient? What is the best way to communicate
these changes in existing documentation?
- What should the implementation/release strategy be? Should this wait until TFF
1.x.x?
- Once changes to current intrinsics have been planned, what qualifies as a
"judicious" use of defaults for maintaining backwards compatiblity?
