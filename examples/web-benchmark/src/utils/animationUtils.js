/**
 * Animation utilities for cycling through parameter combinations
 */

const STEPS_FOR_CONTINUOUS = 10;

/**
 * Generate values for a single parameter
 */
const generateParameterValues = (param) => {
  switch (param.type) {
    case 'slider': {
      const { min, max, step } = param;
      const values = [];

      // Generate STEPS_FOR_CONTINUOUS evenly spaced values
      for (let i = 0; i < STEPS_FOR_CONTINUOUS; i++) {
        const value = min + (max - min) * (i / (STEPS_FOR_CONTINUOUS - 1));

        // Round to step precision if step is defined
        if (step) {
          const rounded = Math.round(value / step) * step;
          values.push(Math.min(max, Math.max(min, rounded)));
        } else {
          values.push(value);
        }
      }

      return values;
    }

    case 'select': {
      // Use all options
      return param.options;
    }

    case 'checkbox': {
      return [false, true];
    }

    case 'color': {
      // Skip color parameters for animation (too complex)
      return [param.default];
    }

    case 'text': {
      // Skip text parameters for animation
      return [param.default];
    }

    default:
      return [param.default];
  }
};

/**
 * Generate cartesian product of arrays
 * e.g. [[1,2], [3,4]] => [[1,3], [1,4], [2,3], [2,4]]
 */
const cartesianProduct = (arrays) => {
  if (arrays.length === 0) return [[]];
  if (arrays.length === 1) return arrays[0].map(v => [v]);

  const [first, ...rest] = arrays;
  const restProduct = cartesianProduct(rest);

  const result = [];
  for (const value of first) {
    for (const combination of restProduct) {
      result.push([value, ...combination]);
    }
  }

  return result;
};

/**
 * Generate all parameter combinations for a demo
 * Returns array of parameter objects, one for each combination
 */
export const generateParameterCombinations = (demo) => {
  if (!demo.params || demo.params.length === 0) {
    return [{}]; // Single empty combination if no params
  }

  // Generate values for each parameter
  const parameterValues = demo.params.map(param => ({
    id: param.id,
    values: generateParameterValues(param)
  }));

  // Get all combinations
  const valueArrays = parameterValues.map(p => p.values);
  const combinations = cartesianProduct(valueArrays);

  // Convert back to param objects
  return combinations.map(combination => {
    const params = {};
    parameterValues.forEach((param, i) => {
      params[param.id] = combination[i];
    });
    return params;
  });
};

/**
 * Create a looping animation sequence (forward and backward)
 */
export const createLoopingSequence = (combinations) => {
  if (combinations.length <= 1) return combinations;

  // Forward + backward (excluding duplicates at ends)
  return [...combinations, ...combinations.slice(1, -1).reverse()];
};

/**
 * Estimate animation duration
 */
export const estimateAnimationDuration = (demo, delayPerFrame = 200) => {
  const combinations = generateParameterCombinations(demo);
  const sequence = createLoopingSequence(combinations);
  const durationMs = sequence.length * delayPerFrame;

  return {
    frames: sequence.length,
    durationMs,
    durationSec: (durationMs / 1000).toFixed(1)
  };
};
