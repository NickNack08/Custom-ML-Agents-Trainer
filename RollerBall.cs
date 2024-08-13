using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerBall : Agent
{
    Rigidbody rBody;
    public float constantSpeed = 5f; // constant speed
    public float platformHeight = 0.5f; // height of your platform
    public int maxSteps = 1000; // Maximum number of steps allowed per episode

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public float previousDistanceToTarget;

    public override void OnEpisodeBegin()
    {
        // Reset the agent's position and momentum
        this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
        this.transform.localPosition = new Vector3(0, platformHeight, 0);

        // Move the target to a new spot
        Target.localPosition = new Vector3(Random.value * 8 - 4,
                                           platformHeight,
                                           Random.value * 8 - 4);

        // Initialize previous distance to target
        previousDistanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition - this.transform.localPosition);
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 0:
                dirToGo = Vector3.forward;  // World space forward
                break;
            case 1:
                dirToGo = Vector3.back;     // World space backward
                break;
            case 2:
                dirToGo = Vector3.left;     // World space left
                break;
            case 3:
                dirToGo = Vector3.right;    // World space right
                break;
        }

        // Set velocity directly
        rBody.velocity = dirToGo.normalized * constantSpeed;

        Debug.Log("Received action: " + act[0]);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);
        
        // Calculate the current distance to the target
        float currentDistanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        // // Reward the agent for reducing distance to the target
        if (currentDistanceToTarget < previousDistanceToTarget)
        {
            AddReward(0.1f); // Reward for getting closer to the target
        }
        else if (currentDistanceToTarget > previousDistanceToTarget)
        {
            AddReward(-0.1f); // Penalize for moving away from the target
        }

        // Update the previous distance to the target
        previousDistanceToTarget = currentDistanceToTarget;

        // Reached target
        if (currentDistanceToTarget < 1.42f)
        {
             // Calculate a time-based bonus
            float timeBonus = 1.0f - (StepCount / (float)maxSteps);
            float finalReward = 10.0f + (10.0f * timeBonus); // Base reward plus time bonus
            SetReward(finalReward);
            EndEpisode();
            EndEpisode();
        }

        if (transform.position.y < platformHeight - 0.1f) // Small threshold to account for physics
        {
            SetReward(-5f); // Set the reward to -5
            EndEpisode(); // End the episode
        }

        // Check if the agent has taken too many steps
        if (StepCount >= maxSteps)
        {
            SetReward(-5f); // Set the reward to -5
            EndEpisode(); // End the episode
        }
    }


    // CONTINUOUS ACTION
    // public override void OnActionReceived(ActionBuffers actionBuffers)
    // {
    //     // Actions, size = 2
    //     Vector3 controlSignal = Vector3.zero;
    //     controlSignal.x = actionBuffers.ContinuousActionsActions[0];
    //     controlSignal.z = actionBuffers.ContinuousActions[1];
    //     rBody.AddForce(controlSignal * forceMultiplier); // adding speed


    //     // Rewards
    //     float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

    //     // Reached target
    //     if (distanceToTarget < 1.42f)
    //     {
    //         SetReward(1.0f); // SetReward is used to finalize a task vs AddReward for intermediate targets
    //         EndEpisode();
    //     }

    //     // Fell off platform
    //     else if (this.transform.localPosition.y < 0)
    //     {
    //         EndEpisode();
    //     }
    // }

    public override void Heuristic(in ActionBuffers actionsOut)
    // heuristic is how you can play the game personally to ensure it works
        {
            var continuousActionsOut = actionsOut.ContinuousActions;
            continuousActionsOut[0] = Input.GetAxis("Horizontal");
            continuousActionsOut[1] = Input.GetAxis("Vertical");
        }
 }