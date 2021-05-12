class StringUtil:
    @staticmethod
    def get_model_type(model_name: str) -> str:
        if "imgtext" in model_name.lower():
            return "image+text"
        elif "img" in model_name.lower():
            return "image"
        return "text"

    @staticmethod
    def get_checkpoint_model_arch(
        model_name: str, model_arch: str, bert_model_arch: str
    ) -> str:
        model_type = StringUtil.get_model_type(model_name)
        if model_type == "image":
            return model_arch
        return bert_model_arch.replace("/", "_")

    @staticmethod
    def str_to_list(s: str):
        return (
            s.replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace("'", "")
            .replace(",", "")
            .split()
        )
