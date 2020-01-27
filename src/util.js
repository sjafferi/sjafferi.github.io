export const groupBy = (key) => (array) =>
  array.reduce(
    (objectsByKeyValue, obj) => ({
      ...objectsByKeyValue,
      [obj[key]]: (objectsByKeyValue[obj[key]] || []).concat(obj)
    }),
    {}
  );

export function toSlug(str) {
  return str.toLowerCase().replace(/[^a-zA-Z\d\s:]/g, "").replace(/ /g, "-")
}
